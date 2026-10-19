"""The one number rule for the harness work: on a store with no harness record on any run, every
`recommend --json` and `posterior --json` field equals the base release's exactly.

Each tree (the base, from `git archive $LOOPMATH_BASE_REF src`, and this one) fits its own copy of the store
and answers in a subprocess with a temp HOME, LOOPMATH_HOME and LOOPMATH_CACHE_DIR. Ids minted per call
(rec_, tsk_, fit_), clock times and timings are blanked before the comparison; nothing else is. The stores:
a fresh one (the shipped prior only) and a temp copy of `$LOOPMATH_COMPARE_STORE` when it is set. Without
LOOPMATH_BASE_REF (a commit of this repository) the test skips, as it does in the public tree.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE_REF = os.environ.get("LOOPMATH_BASE_REF")
OTHER_STORE = os.environ.get("LOOPMATH_COMPARE_STORE")
MINTED = re.compile(r"^((rec|tsk|fit|run)_[0-9A-HJKMNP-TV-Z]{26}|fit_\d{14})$")
VOLATILE_KEYS = frozenset({"created_at", "generated_at", "at", "age_s", "seconds", "fit_time_s"})
HARNESS_KEYS = ("dev.loopmath.harness", "harness")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


@pytest.fixture(scope="module")
def base_src(tmp_path_factory) -> Path:
    if not BASE_REF or shutil.which("git") is None or _git("cat-file", "-e", f"{BASE_REF}^{{commit}}").returncode:
        pytest.skip("set LOOPMATH_BASE_REF to the base commit (a development checkout)")
    out = tmp_path_factory.mktemp("base-tree")
    data = _git("archive", BASE_REF, "src").stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")
    return out / "src"


def _lm(src: Path, home: Path, *argv: str) -> dict:
    tmp = home.parent
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LOOPMATH_", "PYTHON"))}
    env.update(PYTHONPATH=str(src), HOME=str(tmp / "userhome"), LOOPMATH_HOME=str(home),
               LOOPMATH_CACHE_DIR=str(tmp / f"cache-{home.name}"), MPLCONFIGDIR=str(tmp / "mpl"),
               NO_COLOR="1", PYTHON_COLORS="0")
    (tmp / "userhome").mkdir(exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "loopmath", *argv, "--json"], env=env, capture_output=True,
                       text=True, cwd=str(tmp), timeout=900)
    assert p.returncode == 0, (argv, p.stderr[-2000:])
    return json.loads(p.stdout)


def _blank(x):
    if isinstance(x, dict):
        return {k: (None if k in VOLATILE_KEYS else _blank(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_blank(v) for v in x]
    if isinstance(x, str) and MINTED.match(x):
        return x[:4] + "minted"
    return x


def _diff(a, b, path="") -> list[str]:
    """The paths where `a` and `b` differ, for a readable failure."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b), key=str)
                for d in ([f"{path}/{k}: only one side"] if (k in a) != (k in b) else _diff(a[k], b[k], f"{path}/{k}"))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {str(a)[:80]} != {str(b)[:80]}"]


def _no_harness(store: Path) -> None:
    for f in (store / "runs").glob("*.ocp.json"):
        ext = (json.loads(f.read_text()).get("run") or {}).get("ext") or {}
        assert not any(k in ext for k in HARNESS_KEYS), f.name


def _answers(src: Path, store: Path, task: tuple[str, str]) -> dict:
    _lm(src, store, "fit")
    return {"recommend": _blank(_lm(src, store, "recommend", "--type", task[0], "--repo", task[1])),
            "posterior": _blank(_lm(src, store, "posterior"))}


def _compare(base_src: Path, store: Path, tmp_path: Path, task: tuple[str, str]) -> None:
    old, new = tmp_path / "old" / "home", tmp_path / "new" / "home"
    for d in (old, new):
        d.parent.mkdir(parents=True)
        if store.is_dir():
            shutil.copytree(store, d, symlinks=True)
        else:
            d.mkdir()
    _no_harness(new)
    a = _answers(base_src, old, task)
    b = _answers(ROOT / "src", new, task)
    assert _diff(a["recommend"], b["recommend"]) == []
    assert _diff(a["posterior"], b["posterior"]) == []


def test_a_fresh_store_answers_as_the_base(base_src, tmp_path):
    _compare(base_src, tmp_path / "none", tmp_path, ("feature", "acme/app"))


def test_a_copy_of_another_store_answers_as_the_base(base_src, tmp_path):
    if not OTHER_STORE or not (Path(OTHER_STORE) / "runs").is_dir():
        pytest.skip("set LOOPMATH_COMPARE_STORE to a store to copy")
    store = Path(OTHER_STORE)
    rows = [json.loads(ln) for ln in (store / "runs" / "index.jsonl").read_text().splitlines() if ln.strip()]
    first = next(r for r in rows if r.get("task_type") and r.get("repo"))
    _compare(base_src, store, tmp_path, (first["task_type"], first["repo"]))
