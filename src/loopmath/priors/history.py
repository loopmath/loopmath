"""Repo-history task miner (lane 11): our own commits as tasks with tests.

A candidate is a non-merge commit that changes at most `max_files` files and
at least one runnable test file. It becomes one spec 03 `Task` line (D14):
`source: "repo_history"`, `base_commit` = the parent, features from the diff,
and a top-level `history` object with the commit, its test files, the command
that runs them, the changed lines and the files changed.

- Type: the conventional-commit prefix when there is one (`fix:` bug_fix,
  `feat:` feature, `refactor:` refactor, `test:` tests, `chore:`/`build:`/`ci:`
  infra), else `tests` when no source file changed, else subject keywords,
  else feature. The rule that fired is kept in `history.type_rule`; the label
  is `rule:history/1`, not a person's or a model's.
- Size: `changed_lines` counts added plus deleted lines in source files (the
  part an implementer writes when the target's tests are the check); test
  lines are kept apart in `history.test_lines`. For a `tests` task the test
  lines are the size.
- `validate_task` runs the target's test files in a `git archive` export of
  the commit and of its parent with the commit's test files laid over it
  (never a worktree of the source repo), and records `fail_to_pass` and
  `pass_to_pass`. pytest and vitest run (vitest with the source repo's installed
  packages linked in); other runners are `not_run`.

Git is only read (`log`, `cat-file`, `archive`). Run as
`python -m loopmath.priors.history mine|validate|select ...`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from ..taskmodel import size_bucket, touches_bucket

MINER_VERSION = "history/1"
LABEL = "rule:history/1"
_EM = chr(0x2014)
_EM_RUN = re.compile(r"\s*" + _EM + r"\s*")


def plain_title(text: str) -> str:
    """A commit subject as a task title, with any em dash made a comma (house rule; titles are prose)."""
    return _EM_RUN.sub(", ", text)

LANGS = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".rs": "rust", ".go": "go", ".swift": "swift", ".svelte": "svelte", ".vue": "vue",
    ".sh": "shell", ".sql": "sql", ".css": "css", ".scss": "css", ".html": "html",
}
_SKIP = re.compile(
    r"(^|/)(node_modules|dist|build|target|out|vendor|\.venv|__pycache__|coverage|test-results)/"
    r"|(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb?|Cargo\.lock|poetry\.lock|uv\.lock)$"
    r"|\.(snap|min\.js|map|png|jpe?g|gif|ico|svg|webp|pdf|woff2?|ttf|zip|gz)$", re.I)
_TEST_DIR = re.compile(r"(^|/)(tests?|__tests__|e2e)/", re.I)
_TEST_NAME = re.compile(r"(^|/)(test_[^/]+\.py|[^/]+_test\.(py|go|rs)|[^/]+\.(test|spec)\.[cm]?[jt]sx?)$", re.I)
_DOCS = re.compile(r"\.(md|rst|txt|adoc)$|(^|/)docs?/", re.I)
_CONVENTIONAL = re.compile(r"^(?P<kind>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?!?:\s")
_PREFIX_TYPES = {
    "feat": "feature", "feature": "feature", "fix": "bug_fix", "bugfix": "bug_fix", "hotfix": "bug_fix",
    "refactor": "refactor", "style": "refactor", "perf": "refactor", "test": "tests", "tests": "tests",
    "docs": "docs", "doc": "docs", "chore": "infra", "build": "infra", "ci": "infra", "deps": "infra",
}
_KEYWORDS = (
    ("bug_fix", re.compile(r"\b(fix(es|ed)?|bug|repair|correct(s|ed)?|restore|regression|crash(es)?|broken|"
                           r"prevent|stabili[sz]e|hang|leak)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|rename|extract|split|simplif(y|ies)|clean ?up|reorgani[sz]e|dedupe|"
                            r"consolidate|inline|tidy|move)\b", re.I)),
    ("tests", re.compile(r"^(add|write|cover|extend)\b.*\b(tests?|coverage)\b|^tests?\b", re.I)),
    ("infra", re.compile(r"\b(ci|bump|dependenc(y|ies)|tooling|lint(er)?|packaging|workflow file)\b", re.I)),
)
_EXCLUDE_SUBJECT = re.compile(r"agent output|harness commit|^wip\b|^revert\b|^merge\b|^release v?\d|^v\d+\.\d+",
                              re.I)


# ---------------------------------------------------------------- git reading
def _git(repo: Path, *args: str, text: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return out.stdout.decode("utf-8", "replace") if text else out.stdout  # type: ignore[return-value]


def _clean_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode(
            "utf-8", "replace")
    if " => " in path:  # a rename: keep the new path
        if "{" in path:
            path = re.sub(r"\{([^{}]*) => ([^{}]*)\}", r"\2", path)
            path = re.sub(r"//+", "/", path)
        else:
            path = path.split(" => ", 1)[1]
    return path


def iter_commits(repo: Path, ref: str = "HEAD") -> Iterator[dict]:
    """Non-merge commits reachable from `ref`, newest first, with per-file numstat."""
    out = _git(repo, "log", "--no-merges", "-M", "--numstat", "--format=%x1e%H%x1f%P%x1f%aI%x1f%s", ref)
    for block in out.split("\x1e")[1:]:
        head, _, body = block.partition("\n")
        sha, parents, date, subject = (head.split("\x1f") + ["", "", ""])[:4]
        files = []
        for line in body.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            binary = added == "-" or deleted == "-"
            files.append({"path": _clean_path(path), "added": 0 if binary else int(added),
                          "deleted": 0 if binary else int(deleted), "binary": binary})
        yield {"sha": sha, "parents": parents.split(), "date": date, "subject": subject.strip(), "files": files}


class _Blobs:
    """`git cat-file --batch` on one repo: existence and contents of `<commit>:<path>`."""

    def __init__(self, repo: Path):
        self.proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.cache: dict[str, bytes | None] = {}

    def get(self, commit: str, path: str) -> bytes | None:
        key = f"{commit}:{path}"
        if key in self.cache:
            return self.cache[key]
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(key.encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode().split()
        data = None
        if len(header) == 3 and header[1] == "blob":
            data = self.proc.stdout.read(int(header[2]))
            self.proc.stdout.read(1)
        elif len(header) == 3:  # a tree or other object: skip its body
            self.proc.stdout.read(int(header[2]) + 1)
        self.cache[key] = data
        return data

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


# ---------------------------------------------------------------- classification
def is_test_file(path: str) -> bool:
    """A file of the test suite (test code, fixtures, conftest)."""
    return bool(_TEST_DIR.search(path) or _TEST_NAME.search(path))


def is_runnable_test(path: str) -> bool:
    """A test file a runner collects on its own."""
    p = PurePosixPath(path)
    if p.suffix == ".py":
        return p.name.startswith("test_") or p.name.endswith("_test.py")
    if p.suffix == ".rs":
        return bool(_TEST_DIR.search(path)) and p.parent.name == "tests"
    if p.suffix == ".go":
        return p.name.endswith("_test.go")
    if p.suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"):
        return bool(re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", p.name)) or "__tests__" in p.parts
    return False


def classify_files(files: list[dict]) -> dict:
    kept = [f for f in files if not _SKIP.search(f["path"]) and not f["binary"]]
    tests = [f for f in kept if is_test_file(f["path"])]
    source = [f for f in kept if f not in tests and PurePosixPath(f["path"]).suffix.lower() in LANGS
              and not _DOCS.search(f["path"])]
    other = [f for f in kept if f not in tests and f not in source]
    return {"kept": kept, "tests": tests, "runnable": [f["path"] for f in tests if is_runnable_test(f["path"])],
            "source": source, "other": other}


def _lines(files: list[dict]) -> int:
    return sum(f["added"] + f["deleted"] for f in files)


def task_type(subject: str, groups: dict) -> tuple[str, str]:
    """(type, rule) for a commit."""
    m = _CONVENTIONAL.match(subject)
    if m and m.group("kind").lower() in _PREFIX_TYPES:
        kind = _PREFIX_TYPES[m.group("kind").lower()]
        if kind == "docs" and groups["source"]:
            kind = "feature" if not groups["tests"] else kind
        return kind, f"prefix:{m.group('kind').lower()}"
    if not groups["source"]:
        return "tests", "tests_only"
    for kind, pattern in _KEYWORDS:
        if pattern.search(subject):
            return kind, f"keyword:{kind}"
    return "feature", "default"


def _lang(files: list[dict]) -> str | None:
    by = Counter()
    for f in files:
        lang = LANGS.get(PurePosixPath(f["path"]).suffix.lower())
        if lang:
            by[lang] += f["added"] + f["deleted"] or 1
    return by.most_common(1)[0][0] if by else None


def _subtype(subject: str, source: list[dict], tests: list[dict]) -> str | None:
    m = _CONVENTIONAL.match(subject)
    if m and m.group("scope"):
        scope = m.group("scope").split(",")[0].strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,40}", scope):
            return scope
    dirs = Counter(PurePosixPath(f["path"]).parent.name for f in (source or tests) if PurePosixPath(f["path"]).parent.name)
    return dirs.most_common(1)[0][0].lower() if dirs else None


# ---------------------------------------------------------------- test commands
_PY_MARKERS = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py")


def _nearest(blobs: _Blobs, commit: str, path: str, markers: Iterable[str]) -> tuple[str, str] | None:
    """(dir, marker) of the nearest ancestor of `path` holding one of `markers` at `commit`."""
    parts = PurePosixPath(path).parent.parts
    for i in range(len(parts), -1, -1):
        d = "/".join(parts[:i])
        for marker in markers:
            if blobs.get(commit, f"{d}/{marker}" if d else marker) is not None:
                return d or ".", marker
    return None


_RUNNER_NAMES = {"npx vitest run": "vitest", "npx jest": "jest", "npx playwright test": "playwright",
                 "bun test": "bun", "node --test": "node", "npm test --": "npm"}


def _node_runner(package_json: bytes | None, path: str = "") -> str:
    try:
        pkg = json.loads(package_json or b"{}")
    except ValueError:
        pkg = {}
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    script = str((pkg.get("scripts") or {}).get("test") or "")
    if "@playwright/test" in deps and "e2e" in PurePosixPath(path).parts:
        return "npx playwright test"
    if "vitest" in deps or "vitest" in script:
        return "npx vitest run"
    if "jest" in deps or "jest" in script:
        return "npx jest"
    if "bun test" in script:
        return "bun test"
    if "node --test" in script or "tsx --test" in script:
        return "node --test"
    return "npm test --"


def test_command(blobs: _Blobs, commit: str, test_files: list[str]) -> tuple[str, str]:
    """(command, runner) that runs `test_files` at `commit`, from the nearest manifest."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(test_files):
        suffix = PurePosixPath(path).suffix
        if suffix == ".py":
            near = _nearest(blobs, commit, path, _PY_MARKERS)
            groups[("pytest", near[0] if near else ".")].append(path)
        elif suffix == ".rs":
            near = _nearest(blobs, commit, path, ("Cargo.toml",))
            groups[("cargo", near[0] if near else ".")].append(path)
        elif suffix == ".go":
            near = _nearest(blobs, commit, path, ("go.mod",))
            groups[("go", near[0] if near else ".")].append(path)
        else:
            near = _nearest(blobs, commit, path, ("package.json",))
            d = near[0] if near else "."
            runner = _node_runner(blobs.get(commit, f"{d}/package.json" if d != "." else "package.json"), path)
            groups[(runner, d)].append(path)
    parts = []
    runners = []
    for (runner, d), files in sorted(groups.items()):
        rel = [str(PurePosixPath(f).relative_to(d)) if d != "." else f for f in files]
        if runner == "pytest":
            cmd = "python -m pytest -q " + " ".join(rel)
        elif runner == "cargo":
            cmd = "cargo test " + " ".join(f"--test {PurePosixPath(f).stem}" for f in rel)
        elif runner == "go":
            cmd = "go test " + " ".join(sorted({"./" + str(PurePosixPath(f).parent) for f in rel}))
        else:
            cmd = f"{runner} " + " ".join(rel)
        parts.append(cmd if d == "." else f"cd {d} && {cmd}")
        runners.append(_RUNNER_NAMES.get(runner, runner))
    return " && ".join(f"({p})" if len(parts) > 1 else p for p in parts), "+".join(sorted(set(runners)))


# ---------------------------------------------------------------- mining
def commit_to_task(repo_path: Path, name: str, commit: dict, blobs: _Blobs, *, max_files: int = 25) -> dict | None:
    """One D14 task line for a commit, or None when it is not a candidate."""
    if len(commit["parents"]) != 1 or _EXCLUDE_SUBJECT.search(commit["subject"]):
        return None
    groups = classify_files(commit["files"])
    if not groups["runnable"] or not groups["kept"] or len(commit["files"]) > max_files:
        return None
    kind, rule = task_type(commit["subject"], groups)
    if kind == "docs":
        return None
    work = groups["tests"] if kind == "tests" else groups["source"]
    if not work:
        return None
    changed = _lines(work)
    if changed == 0:
        return None
    command, runner = test_command(blobs, commit["sha"], groups["runnable"])
    features = {
        "size": size_bucket(changed),
        "has_tests": "yes",
        "touches": touches_bucket(len(work)),
    }
    lang = _lang(work)
    if lang:
        features["lang"] = lang
    sha = commit["sha"]
    return {
        "id": f"{name}@{sha[:12]}",
        "type": kind,
        "repo": name,
        "title": plain_title(commit["subject"]),
        "subtype": _subtype(commit["subject"], groups["source"], groups["tests"]),
        "features": dict(sorted(features.items())),
        "base_commit": commit["parents"][0],
        "source": "repo_history",
        "labeled_by": LABEL,
        "history": {
            "repo_path": str(repo_path),
            "commit": sha,
            "parent": commit["parents"][0],
            "date": commit["date"],
            "test_files": sorted(groups["runnable"]),
            "test_support_files": sorted(f["path"] for f in groups["tests"] if f["path"] not in groups["runnable"]),
            "test_cmd": command,
            "runner": runner,
            "changed_lines": changed,
            "test_lines": _lines(groups["tests"]),
            "files_changed": len(commit["files"]),
            "source_files": sorted(f["path"] for f in groups["source"]),
            "type_rule": rule,
            "miner": MINER_VERSION,
        },
    }


def mine_repo(repo_path: Path, *, name: str | None = None, ref: str = "HEAD", max_files: int = 25) -> Iterator[dict]:
    """Candidate task lines from one repo, newest first."""
    repo_path = Path(repo_path).expanduser().resolve()
    name = name or repo_path.name
    blobs = _Blobs(repo_path)
    try:
        for commit in iter_commits(repo_path, ref):
            task = commit_to_task(repo_path, name, commit, blobs, max_files=max_files)
            if task:
                yield task
    finally:
        blobs.close()


# ---------------------------------------------------------------- validation
def _export(repo: Path, commit: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True,
                             check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def _junit(path: Path) -> dict[str, str]:
    """Test id to pass | fail | skip from a pytest junit file."""
    if not path.is_file():
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    out: dict[str, str] = {}
    for case in root.iter("testcase"):
        tid = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None or case.find("error") is not None:
            out[tid] = "fail"
        elif case.find("skipped") is not None:
            out[tid] = "skip"
        else:
            out[tid] = "pass"
    return out


def _run_pytest(root: Path, test_files: list[str], *, python: str, timeout: int, scratch: Path) -> tuple[dict, str]:
    """Run the pytest groups of a task in an exported tree; (results, status)."""
    results: dict[str, str] = {}
    by_dir: dict[str, list[str]] = defaultdict(list)
    for f in test_files:
        if not f.endswith(".py"):
            continue
        d = "."
        parts = PurePosixPath(f).parent.parts
        for i in range(len(parts), -1, -1):
            cand = "/".join(parts[:i]) or "."
            if any((root / cand / m).is_file() for m in _PY_MARKERS):
                d = cand
                break
        by_dir[d].append(f)
    status = "ok"
    for n, (d, files) in enumerate(sorted(by_dir.items())):
        cwd = root / d
        present = [str(PurePosixPath(f).relative_to(d)) if d != "." else f for f in files if (root / f).is_file()]
        if not present:
            continue
        home = scratch / "home"
        home.mkdir(exist_ok=True)
        src = cwd / "src"
        env = {
            "PATH": f"{Path(python).parent}:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "HOME": str(home), "TMPDIR": str(scratch), "LANG": "en_US.UTF-8",
            "PYTHONPATH": str(src if src.is_dir() else cwd), "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1", "PYTHON_COLORS": "0", "LOOPMATH_HOME": str(home / "loopmath"),
            "DAGR_HOME": str(home / "dagr"),
        }
        junit = scratch / f"junit-{root.name}-{n}.xml"
        try:
            subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *present],
                           cwd=cwd, env=env, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        results.update(_junit(junit))
    return results, status


def _link_node_modules(source: Path, dest: Path) -> bool:
    """A real `node_modules` folder in `dest` whose entries link to `source`'s packages.

    Tools write caches under `node_modules/.vite`; with a real folder those land
    in the temporary tree, never in the source repo.
    """
    if not source.is_dir():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name in (".vite", ".vitest", ".cache"):
            continue
        link = dest / entry.name
        if not link.exists():
            link.symlink_to(entry)
    return True


def _run_vitest(root: Path, test_files: list[str], *, repo: Path, timeout: int, scratch: Path) -> tuple[dict, str]:
    """Run the vitest groups of a task in an exported tree, with the source repo's installed packages."""
    results: dict[str, str] = {}
    by_dir: dict[str, list[str]] = defaultdict(list)
    for f in test_files:
        parts = PurePosixPath(f).parent.parts
        d = "."
        for i in range(len(parts), -1, -1):
            cand = "/".join(parts[:i]) or "."
            if (root / cand / "package.json").is_file():
                d = cand
                break
        by_dir[d].append(f)
    status = "ok"
    for n, (d, files) in enumerate(sorted(by_dir.items())):
        cwd = root / d
        if not _link_node_modules(repo / d / "node_modules", cwd / "node_modules"):
            return results, "no_toolchain"
        if d != ".":
            _link_node_modules(repo / "node_modules", root / "node_modules")
        vitest = cwd / "node_modules" / ".bin" / "vitest"
        if not vitest.exists():
            return results, "no_toolchain"
        present = [str(PurePosixPath(f).relative_to(d)) if d != "." else f for f in files if (root / f).is_file()]
        if not present:
            continue
        home = scratch / "home"
        home.mkdir(exist_ok=True)
        node_bin = shutil.which("node") or "/usr/local/bin/node"
        env = {"PATH": f"{Path(node_bin).parent}:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
               "HOME": str(home), "TMPDIR": str(scratch), "LANG": "en_US.UTF-8", "CI": "1", "NO_COLOR": "1",
               "FORCE_COLOR": "0"}
        junit = scratch / f"junit-{root.name}-{n}.xml"
        try:
            subprocess.run([str(vitest), "run", *present, "--reporter=junit", f"--outputFile={junit}"],
                           cwd=cwd, env=env, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        results.update(_junit(junit))
    return results, status


def validate_task(task: dict, *, python: str = sys.executable, timeout: int = 240) -> dict:
    """Run the task's test files after and before the commit; returns `history.validation`."""
    h = task["history"]
    runner = h.get("runner")
    if runner not in ("pytest", "vitest"):
        return {"status": "not_run", "reason": f"runner {runner} is not run by the validator yet"}
    repo = Path(h["repo_path"])
    with tempfile.TemporaryDirectory(prefix="lm-hist-") as tmp:
        base = Path(tmp)
        after, before = base / "after", base / "before"
        try:
            _export(repo, h["commit"], after)
            _export(repo, h["parent"], before)
        except subprocess.CalledProcessError as exc:
            return {"status": "error", "reason": f"git archive failed: {exc.returncode}"}
        for rel in h["test_files"] + h.get("test_support_files", []):  # the commit's tests over the parent
            src = after / rel
            if src.is_file():
                (before / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, before / rel)
        if runner == "pytest":
            after_res, after_status = _run_pytest(after, h["test_files"], python=python, timeout=timeout,
                                                  scratch=base)
            before_res, before_status = _run_pytest(before, h["test_files"], python=python, timeout=timeout,
                                                    scratch=base)
        else:
            after_res, after_status = _run_vitest(after, h["test_files"], repo=repo, timeout=timeout, scratch=base)
            before_res, before_status = _run_vitest(before, h["test_files"], repo=repo, timeout=timeout,
                                                    scratch=base)
    passed_after = sorted(t for t, v in after_res.items() if v == "pass")
    failed_after = sorted(t for t, v in after_res.items() if v == "fail")
    f2p = sorted(t for t in passed_after if before_res.get(t) != "pass")
    p2p = sorted(t for t in passed_after if before_res.get(t) == "pass")
    if "no_toolchain" in (after_status, before_status):
        return {"status": "not_run", "reason": f"no installed {runner} in the source repo"}
    if after_status == "timeout" or before_status == "timeout":
        status = "timeout"
    elif not after_res:
        status = "error"
    elif failed_after:
        status = "after_fails"
    elif f2p or task["type"] == "tests":
        status = "verified"
    else:
        status = "no_fail_to_pass"
    out = {"status": status, "fail_to_pass": f2p, "pass_to_pass": p2p, "after_failures": len(failed_after),
           "runner": runner}
    if runner == "pytest":
        out["python"] = Path(python).name
    return out


# ---------------------------------------------------------------- selection
def _rank(task: dict) -> str:
    return hashlib.sha256(task["id"].encode()).hexdigest()


def can_fail(task: dict) -> bool:
    """D66: a task has a test that fails at its base commit (a test-only commit whose parent already passes has none)."""
    return bool((task["history"].get("validation") or {}).get("fail_to_pass"))


def select_tasks(tasks: list[dict], *, per_repo: dict[str, int], require_verified: Iterable[str] = (),
                 dropped: list[dict] | None = None) -> list[dict]:
    """A deterministic, type-balanced pick: round robin over types within each repo, ranked by id hash.

    In a repo held to verified tasks, a picked task that cannot fail is left out after the pick and
    appended to `dropped` (D66: its start tree already passes); the rest of the pick does not move.
    """
    strict = set(require_verified)
    picked: list[dict] = []
    for repo, quota in per_repo.items():
        pool = [t for t in tasks if t["repo"] == repo and (
            repo not in strict or (t["history"].get("validation") or {}).get("status") == "verified")]
        pool = [t for t in pool if t["features"].get("size") != "xl"]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for t in sorted(pool, key=_rank):
            by_type[t["type"]].append(t)
        order = sorted(by_type, key=lambda k: (-len(by_type[k]), k))
        taken: list[dict] = []
        while len(taken) < quota and any(by_type.values()):
            for kind in order:
                if by_type[kind] and len(taken) < quota:
                    taken.append(by_type[kind].pop(0))
        taken.sort(key=lambda t: t["history"]["date"])
        if repo in strict:
            if dropped is not None:
                dropped.extend(t for t in taken if not can_fail(t))
            taken = [t for t in taken if can_fail(t)]
        picked.extend({**t, "title": plain_title(t["title"])} for t in taken)
    return picked


# ---------------------------------------------------------------- command line
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Test ids keep their exact text (runners match them); an em dash in them is written as its
            # JSON escape, so the file holds none and a JSON reader gets the same string back.
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False).replace(_EM, "\\u2014") + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def _existing_file(value: str) -> str:
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"no such file: {value}")
    return value


def _quota(value: str) -> tuple[str, int]:
    repo, _, n = value.partition("=")
    if not repo or not n.isdigit() or int(n) < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is not REPO=N with N at least 1")
    return repo, int(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loopmath.priors.history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="candidate tasks from one repo")
    m.add_argument("repo")
    m.add_argument("--name")
    m.add_argument("--ref", default="HEAD")
    m.add_argument("--max-files", type=int, default=25)
    m.add_argument("--out", required=True)
    v = sub.add_parser("validate", help="run each task's tests after and before its commit")
    v.add_argument("infile", type=_existing_file)
    v.add_argument("--out", required=True)
    v.add_argument("--python", default=sys.executable)
    v.add_argument("--workers", type=int, default=3)
    v.add_argument("--timeout", type=int, default=240)
    v.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("select", help="pick the task set")
    s.add_argument("infiles", nargs="+", type=_existing_file, help="validated candidate files (JSON Lines)")
    s.add_argument("--quota", action="append", default=[], type=_quota, metavar="REPO=N",
                   help="at most N tasks from REPO; repeat per repo (required)")
    s.add_argument("--verified", action="append", default=[], metavar="REPO",
                   help="REPO keeps verified tasks that can fail only (D66); repeat per repo")
    s.add_argument("--out", required=True, help="the task set to write (JSON Lines)")
    args = ap.parse_args(argv)

    if args.cmd == "mine":
        rows = list(mine_repo(Path(args.repo), name=args.name, ref=args.ref, max_files=args.max_files))
        _write_jsonl(Path(args.out), rows)
        print(f"{len(rows)} candidates; types {dict(Counter(r['type'] for r in rows))}")
        return 0
    if args.cmd == "validate":
        rows = _read_jsonl(Path(args.infile))
        todo = rows[: args.limit] if args.limit else rows

        def one(task: dict) -> dict:
            task["history"]["validation"] = validate_task(task, python=args.python, timeout=args.timeout)
            print(f"{task['id']} {task['history']['validation']['status']}", file=sys.stderr, flush=True)
            return task

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            done = list(pool.map(one, todo))
        _write_jsonl(Path(args.out), done)
        print(f"{len(done)} validated; {dict(Counter(t['history']['validation']['status'] for t in done))}")
        return 0
    rows = [r for f in args.infiles for r in _read_jsonl(Path(f))]
    quota = dict(args.quota)
    if not quota:
        s.error("select needs --quota REPO=N for each repo to pick from")
    # A misspelt repo would silently pick nothing from it, or skip its D66 filter: refuse, write nothing.
    held = {r.get("repo") for r in rows}
    unknown = sorted((set(quota) | set(args.verified)) - held)
    if unknown:
        s.error(f"no tasks from {', '.join(unknown)} in the input files; they hold "
                 f"{', '.join(sorted(str(r) for r in held)) or 'no tasks'}")
    no_quota = sorted(set(args.verified) - set(quota))
    if no_quota:
        s.error(f"--verified {', '.join(no_quota)} has no --quota")
    dropped: list[dict] = []
    picked = select_tasks(rows, per_repo=quota, require_verified=args.verified, dropped=dropped)
    if not picked:
        print(f"no tasks picked (none verified and able to fail within the quotas); {args.out} not written",
              file=sys.stderr)
        return 1
    _write_jsonl(Path(args.out), picked)
    print(f"wrote {len(picked)} tasks to {args.out}; types {dict(Counter(t['type'] for t in picked))}; "
          f"repos {dict(Counter(t['repo'] for t in picked))}")
    if dropped:
        print(f"left out {len(dropped)} that cannot fail (D66): {' '.join(t['id'] for t in dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
