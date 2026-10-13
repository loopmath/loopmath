"""Artifact classification and heredoc-title recovery."""

from __future__ import annotations

import re
from pathlib import PurePath

KINDS = ("plan", "review", "spec", "report", "code", "test", "config", "doc", "data", "log", "other")
TEST_RE = re.compile(r"(^|/)tests?/|(^|/)(test_[^/]*\.py|[^/]*_test\.[a-z]+|conftest\.py)$")
DOC_EXT = {"md", "rst", "txt", "adoc", "markdown"}
CODE_EXT = {"py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "rs", "go", "java", "kt", "scala", "c", "cc", "cpp", "h", "hpp", "rb", "sh", "bash", "zsh", "fish", "ps1", "sql", "swift", "lua", "pl", "php", "cs", "m", "mm", "r", "jl", "ex", "exs", "hs", "el"}
CONFIG_EXT = {"toml", "yaml", "yml", "json", "ini", "cfg", "conf", "lock", "env", "properties", "editorconfig"}
DATA_EXT = {"csv", "tsv", "jsonl", "ndjson", "parquet", "sqlite", "sqlite3", "db", "npy", "npz", "pkl", "pickle", "arrow", "feather", "xlsx", "h5"}
LOG_EXT = {"log", "out", "err"}
LOG_PATH_RE = re.compile(r"(^|/)logs?/|(^|[./_-])logs?([._-]|$)(?=[^/]*$)")
DOC_NAME_RULES = (
    ("plan", re.compile(r"plan|roadmap|todo|backlog|tasks?\b|status", re.I)),
    ("review", re.compile(r"review|verdict|referee|audit|critique", re.I)),
    ("spec", re.compile(r"spec|rfc|contract|requirements|amendment", re.I)),
    ("report", re.compile(r"report|recon|summary|findings|postmortem|retro", re.I)),
)
HINT_RULES = (
    ("plan", re.compile(r"\b(plan|planning|roadmap|todo|backlog|build status)\b", re.I)),
    ("review", re.compile(r"\b(review|verdict|referee|audit|critique|approve[d]?|flags)\b", re.I)),
    ("spec", re.compile(r"\b(spec|specification|rfc|contract|requirements|amendments?)\b", re.I)),
    ("report", re.compile(r"\b(report|recon|summary|findings|postmortem|retro)\b", re.I)),
)
HEADING_RE = re.compile(r"^(#+\s|[A-Z][A-Z -]{2,}:|=+\s)")
MAX_TITLE = 60
HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
MAX_HINT = 200

EXTENSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_+-]*$")


def language_from_path(path: str) -> tuple[str | None, str | None]:
    """Return the lower-case final extension identifier at heuristic tier.

    This field makes no semantic language claim: ``x.H`` yields ``"h"`` and
    ``x.ts`` yields ``"ts"``, not C or TypeScript.  Only the basename is used;
    the identifier is the text after its final dot, ASCII-lowercased.  A leading
    dot alone, an empty suffix, or a suffix outside ``[a-z0-9][a-z0-9_+-]*`` is
    absent/invalid and yields ``(None, None)``.  Thus ``archive.tar.GZ`` is ``gz``.
    """
    name = path.rsplit("/", 1)[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name[1:] else ""
    return (ext, "heuristic") if EXTENSION_ID_RE.fullmatch(ext) else (None, None)


def tests_touched_from_paths(paths: list[str]) -> tuple[int, str]:
    """Count distinct test-file paths by the graph's exact heuristic path rule.

    A path counts when it is below a case-sensitive ``test/`` or ``tests/``
    directory, or its basename is ``test_*.py``, ``*_test.<lowercase letters>``,
    or ``conftest.py``.  Only forward slashes delimit path components.  The count
    is therefore heuristic even when it is zero.
    """
    return len({path for path in paths if TEST_RE.search(path)}), "heuristic"


def doc_subkind(path: str) -> str | None:
    parts = [part for part in PurePath(path).parts if part not in ("/", "")]
    if not parts:
        return None
    for kind, rx in DOC_NAME_RULES:
        if rx.search(parts[-1]):
            return kind
    for kind, rx in DOC_NAME_RULES:
        if any(rx.search(directory) for directory in parts[:-1]):
            return kind
    return None


def kind_from_hint(text_hint: str | None) -> str | None:
    """Return the document kind named by a title-like first line, if any."""
    if not text_hint:
        return None
    first = str(text_hint).strip().splitlines()
    if not first:
        return None
    line = first[0].strip()
    if not (HEADING_RE.match(line) or len(line) <= MAX_TITLE):
        return None
    for kind, rx in HINT_RULES:
        if rx.search(line):
            return kind
    return None


def artifact_kind(path: str, text_hint: str | None = None) -> tuple[str | None, str | None]:
    """Return the heuristic artifact kind for *path*, refined by a title hint."""
    if not path:
        return None, None
    name = path.rsplit("/", 1)[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name[1:] else ""
    if TEST_RE.search(path):
        return "test", "heuristic"
    if ext in LOG_EXT or LOG_PATH_RE.search(path):
        return "log", "heuristic"
    if ext in DOC_EXT or not ext:
        kind = doc_subkind(path) or kind_from_hint(text_hint)
        return kind or ("doc" if ext in DOC_EXT else "other"), "heuristic"
    if ext in CODE_EXT:
        return "code", "heuristic"
    if ext in DATA_EXT:
        return "data", "heuristic"
    if ext in CONFIG_EXT:
        return "config", "heuristic"
    return kind_from_hint(text_hint) or "other", "heuristic"


def heredoc_hints(command: str) -> list[dict]:
    """Recover the first non-empty body line for each heredoc in *command*."""
    lines = command.split("\n")
    out: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        markers = HEREDOC_RE.findall(line)
        index += 1
        for _, word in markers:
            first: str | None = None
            terminated = False
            while index < len(lines):
                body = lines[index]
                index += 1
                if body.rstrip() == word or body.lstrip("\t").rstrip() == word:
                    terminated = True
                    break
                if first is None and body.strip():
                    first = body.strip()[:MAX_HINT]
            item = {"marker": line.strip(), "first_line": first}
            if not terminated:
                item["unterminated"] = True
            out.append(item)
    return out


def hint_for_write(write: dict, bash_entries: list[dict]) -> str | None:
    if "heredoc" not in str(write.get("how") or ""):
        return None
    name = str(write.get("path") or "").rsplit("/", 1)[-1]
    raw = write.get("raw")
    for entry in bash_entries:
        if entry.get("ts") != write.get("ts") or not isinstance(entry.get("command"), str):
            continue
        hints = [hint for hint in heredoc_hints(entry["command"]) if hint["first_line"]]
        if not hints:
            continue
        named = [hint for hint in hints if (name and name in hint["marker"]) or (isinstance(raw, str) and raw and raw in hint["marker"])]
        if len(named) == 1:
            return named[0]["first_line"]
        if not named and len(hints) == 1:
            return hints[0]["first_line"]
    return None
