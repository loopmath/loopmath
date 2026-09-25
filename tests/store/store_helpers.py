"""Shared helpers for the lane 07 store tests (conftest.py holds only the log-folder isolation fixture)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from loopmath.cli import main
from loopmath.types import Configuration, Task

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
SRC = str(Path(__file__).resolve().parents[2] / "src")


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def rec_payload() -> dict[str, Any]:
    """The recommend fixture with every configuration id recomputed by this tree's `config_id`, so the
    tests hold before and after lanes 1 and 4 change the canonical form."""
    from loopmath.workflows.ids import config_id

    text = (FIXTURES / "recommend-binary.json").read_text(encoding="utf-8")
    ids: dict[str, str] = {}
    for node in _walk(json.loads(text)):
        if str(node.get("id", "")).startswith("cfg_") and isinstance(node.get("workflow"), dict) and "settings" in node:
            cfg = Configuration.from_dict(node)
            ids[node["id"]] = config_id(cfg.workflow, cfg.settings)
    for old, new in ids.items():
        text = text.replace(old, new)
    return json.loads(text)


REC_ID = rec_payload()["rec"]
EXPLORE_CFG = rec_payload()["exploration"]["best_value"]["candidate"]["config"]["id"]  # the best-value pick
USUAL_CFG = rec_payload()["usual"]["config"]["id"]


def install_rec(home: Path) -> str:
    (home / "recs").mkdir(parents=True, exist_ok=True)
    (home / "recs" / f"{REC_ID}.json").write_text(json.dumps(rec_payload()), encoding="utf-8")
    return REC_ID


def finished_doc(run: str, *, source: str = "usual", usd: float | None = None, started: str = "2026-09-20T10:00:00-07:00",
                 open_: bool = False) -> dict[str, Any]:
    """A small OCP v0.3 run that passes the strict checker (a stand-in for onboard or an orchestrator)."""
    attempt: dict[str, Any] = {"id": f"att_{run}", "node": "n", "n": 1, "status": "done", "started_at": started,
                               "ended_at": started, "outcome": {"result": "done", "evidence": "reported"}}
    if usd is not None:
        attempt["cost"] = {"usd": usd, "input_tokens": 100, "output_tokens": 10, "basis": "measured"}
    return {"ocp": "0.3", "producer": {"name": "loopmath", "version": "0.1.0"}, "privacy": {"profile": "metadata_only"},
            "run": {"id": run, "started_at": started, "task": {"id": f"tsk_{run}", "type": "docs", "repo": "r"},
                    "configuration": {"id": "cfg_aaaaaaaaaaaa", "source": source}},
            "nodes": [{"id": "n", "kind": "impl"}], "attempts": [] if open_ else [attempt]}


def usual_config() -> Configuration:
    return Configuration.from_dict(rec_payload()["usual"]["config"])


def task(**over: Any) -> Task:
    data = {"id": "tsk_test01", "type": "feature", "repo": "acme/web", "title": "add a flag", "base_commit": None}
    data.update(over)
    return Task.from_dict(data)


def cli(capsys, home: Path, *argv: str) -> tuple[int, Any, str]:
    """Run `loopmath ARGV --home HOME [--json]` in process: (exit code, parsed JSON or text, stderr)."""
    capsys.readouterr()
    code = main([*argv, "--home", str(home)])
    out, err = capsys.readouterr()
    if "--json" in argv and out.strip():
        return code, json.loads(out), err
    return code, out, err


def fake_settle(usd: float = 0.5, tokens: tuple[int, int, int, int] = (1000, 200, 0, 300), basis: str = "measured"):
    """A stand-in for lane 2 `settle_run`: every attempt matched with the given cost."""

    def settle(doc: dict[str, Any]):
        for a in doc.get("attempts") or []:
            a["cost"] = {"input_tokens": tokens[0], "cached_input_tokens": tokens[1],
                         "cache_creation_tokens": tokens[2], "output_tokens": tokens[3], "usd": usd, "basis": basis,
                         "ext": {"dev.loopmath.logmatch": {"tier": "verified" if basis == "measured" else "heuristic"}}}
        return doc, {"unmatched": {}}

    return settle


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_NOSYSTEM": "1"}
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return out.stdout.strip()


def repo_with_two_commits(path: Path) -> tuple[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "a.txt").write_text("one\n")
    git(path, "add", "a.txt")
    git(path, "commit", "-q", "-m", "one")
    first = git(path, "rev-parse", "HEAD")
    (path / "a.txt").write_text("two\n")
    git(path, "commit", "-q", "-am", "two")
    second = git(path, "rev-parse", "HEAD")
    return first, second
