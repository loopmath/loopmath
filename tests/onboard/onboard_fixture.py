"""A synthetic Claude Code and Codex history, and fakes for lanes 4, 5 and 7, for the onboard tests.

Nothing here comes from a real session. The builder writes, under `tmp_path`:

- `work/app`: a git repo whose origin is `git@github.com:acme/app.git` (local only);
- `work/docs`: a git repo with no remote;
- `logs/projects/-work-app/lead-1.jsonl`: a Claude Code root session in `work/app`
  ("Fix the crash ...") that spawns a sub-agent through Task and runs `codex exec`;
- `logs/projects/-work-app/lead-1/subagents/agent-dev.jsonl` (+ `.meta.json`);
- `logs/sessions/rollout-...-review.jsonl`: the Codex session that `codex exec` started;
- `logs/sessions/rollout-...-docs.jsonl`: a lone Codex session in `work/docs`;
- `logs/projects/-work-app/slash-1.jsonl`: a Claude Code session whose only user
  line is a slash command, so it has no prompt to label.

Timestamps sit two days before now, so any window of a few days holds them.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


SECRET_PROMPT = "Fix the crash when the parser reads an empty file SECRET-PROMPT-MARKER"
DOCS_PROMPT = "Add a section to the README about install DOCS-PROMPT-MARKER"


@dataclass
class FixtureHistory:
    root: Path
    logs: Path
    app: Path
    docs: Path
    t0: dt.datetime


def _ts(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _git_repo(path: Path, remote: str | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    if remote:
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote], check=True)


def _claude_turn(sid: str, cwd: Path, t: dt.datetime, req: str, content: list, *, model: str = "claude-opus-5",
                 usage: tuple[int, int, int, int] = (1000, 200, 5000, 300), side: str | None = None) -> dict:
    inp, cw, cr, out = usage
    row = {"type": "assistant", "timestamp": _ts(t), "sessionId": sid, "cwd": str(cwd), "requestId": req,
           "message": {"model": model, "role": "assistant", "content": content, "stop_reason": "tool_use",
                       "usage": {"input_tokens": inp, "cache_creation_input_tokens": cw,
                                 "cache_read_input_tokens": cr, "output_tokens": out}}}
    if side:
        row.update({"isSidechain": True, "agentId": side})
    return row


def _user(sid: str, cwd: Path, t: dt.datetime, content, *, side: str | None = None, meta: bool = False) -> dict:
    row = {"type": "user", "timestamp": _ts(t), "sessionId": sid, "cwd": str(cwd),
           "message": {"role": "user", "content": content}}
    if side:
        row.update({"isSidechain": True, "agentId": side})
    if meta:
        row["isMeta"] = True
    return row


def _codex(path: Path, sid: str, cwd: Path, t: dt.datetime, prompt: str, *, model: str, effort: str,
           originator: str, minutes: float, usage: tuple[int, int, int]) -> None:
    inp, cached, out = usage
    end = t + dt.timedelta(minutes=minutes)
    _write_jsonl(path, [
        {"timestamp": _ts(t), "type": "session_meta",
         "payload": {"id": sid, "timestamp": _ts(t), "cwd": str(cwd), "originator": originator,
                     "cli_version": "0.150.0", "source": "exec" if originator == "codex_exec" else "cli"}},
        {"timestamp": _ts(t), "type": "turn_context",
         "payload": {"turn_id": "t1", "cwd": str(cwd), "model": model, "effort": effort}},
        {"timestamp": _ts(t), "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": f"<environment_context>cwd={cwd}</environment_context>"}]}},
        {"timestamp": _ts(t), "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}},
        {"timestamp": _ts(end), "type": "event_msg",
         "payload": {"type": "token_count", "info": {"total_token_usage": {
             "input_tokens": inp, "cached_input_tokens": cached, "cache_write_input_tokens": 0,
             "output_tokens": out, "reasoning_output_tokens": 0, "total_tokens": inp + out}}}},
        {"timestamp": _ts(end), "type": "response_item",
         "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}},
    ])


def build_history(root: Path, *, now: dt.datetime | None = None) -> FixtureHistory:
    now = now or dt.datetime.now(dt.timezone.utc)
    t0 = (now - dt.timedelta(days=2)).replace(microsecond=0)
    logs = root / "logs"
    app = root / "work" / "app"
    docs = root / "work" / "docs"
    _git_repo(app, "git@github.com:acme/app.git")
    _git_repo(docs, None)

    proj = logs / "projects" / "-work-app"
    m = lambda k: t0 + dt.timedelta(minutes=k)  # noqa: E731
    _write_jsonl(proj / "lead-1.jsonl", [
        _user("lead-1", app, m(0), SECRET_PROMPT),
        _claude_turn("lead-1", app, m(0.5), "r1", [{"type": "tool_use", "id": "toolu_dev", "name": "Task",
                     "input": {"description": "Fix the parser", "subagent_type": "general-purpose", "prompt": "fix it"}}]),
        _user("lead-1", app, m(4), [{"type": "tool_result", "tool_use_id": "toolu_dev", "content": "Done."}]),
        _claude_turn("lead-1", app, m(5), "r2", [{"type": "tool_use", "id": "toolu_b1", "name": "Bash",
                     "input": {"command": "codex exec --model gpt-5.6-sol \"Review the diff\"", "description": "review"}}]),
        _user("lead-1", app, m(9), [{"type": "tool_result", "tool_use_id": "toolu_b1", "content": "reviewed"}]),
        _claude_turn("lead-1", app, m(10), "r3", [{"type": "text", "text": "All done."}]),
    ])
    sub = proj / "lead-1" / "subagents"
    _write_jsonl(sub / "agent-dev.jsonl", [
        _user("lead-1", app, m(1), "fix it", side="agent-dev"),
        _claude_turn("lead-1", app, m(1.5), "s1", [{"type": "tool_use", "id": "toolu_e1", "name": "Edit",
                     "input": {"file_path": str(app / "parser.py"), "old_string": "a", "new_string": "b"}}],
                     model="claude-sonnet-5", side="agent-dev", usage=(800, 100, 3000, 400)),
        _user("lead-1", app, m(2), [{"type": "tool_result", "tool_use_id": "toolu_e1", "content": "ok"}], side="agent-dev"),
        _claude_turn("lead-1", app, m(3.5), "s2", [{"type": "text", "text": "fixed"}],
                     model="claude-sonnet-5", side="agent-dev", usage=(900, 0, 3500, 200)),
    ])
    (sub / "agent-dev.meta.json").write_text(json.dumps(
        {"toolUseId": "toolu_dev", "agentType": "general-purpose", "spawnDepth": 1, "description": "Fix the parser"}))
    _write_jsonl(proj / "slash-1.jsonl", [
        _user("slash-1", app, m(60), "<command-name>/usage</command-name>"),
        _claude_turn("slash-1", app, m(61), "q1", [{"type": "text", "text": "usage shown"}], usage=(10, 0, 0, 5)),
        _claude_turn("slash-1", app, m(62.5), "q2", [{"type": "text", "text": "bye"}], usage=(10, 0, 0, 5)),
    ])

    sessions = logs / "sessions"
    stamp = t0.strftime("%Y-%m-%dT%H-%M-%S")
    _codex(sessions / f"rollout-{stamp}-review.jsonl", "cx-review-1", app, m(5.05), "Review the diff",
           model="gpt-5.6-sol", effort="high", originator="codex_exec", minutes=3, usage=(20000, 12000, 900))
    _codex(sessions / f"rollout-{stamp}-docs.jsonl", "cx-docs-1", docs, m(120), DOCS_PROMPT,
           model="gpt-5.6-terra", effort="medium", originator="codex_cli_rs", minutes=6, usage=(30000, 20000, 1500))
    return FixtureHistory(root=root, logs=logs, app=app, docs=docs, t0=t0)


# ---------------------------------------------------------------- fakes for lanes 4, 5 and 7
def configuration(shape: str, *, node_vertex: dict | None = None):
    """A real `types.Configuration` of the kind lane 4's `infer` returns, on a catalog shape."""
    from loopmath.types import Configuration, Setting
    from loopmath.workflows.format import catalog
    from loopmath.workflows.ids import config_id

    wf = catalog()["implement_review" if shape == "implement_review" else "solo"]
    if shape == "implement_review":
        settings = {"implement": Setting("claude-code", "opus-5"), "review": Setting("codex", "gpt-5.6-sol", "high")}
    else:
        settings = {"implement": Setting("codex", "gpt-5.6-terra", "medium")}
    extra = {"node_vertex": dict(node_vertex)} if node_vertex else {}
    return Configuration(config_id(wf, settings), wf, settings, extra)


def fake_infer(graph):
    """Two or more sessions: implement_review (reviewers on `review`); one session: solo."""
    if len(graph.nodes) > 1:
        vertex = {n.id: ("review" if n.role == "reviewer" else "implement") for n in graph.nodes}
        return configuration("implement_review", node_vertex=vertex), 0.8
    return configuration("solo"), 0.6


class FakeConfig:
    def __init__(self, values: dict | None = None):
        self.values = dict(values or {})
        self.saved = 0

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def save(self):
        self.saved += 1


class FakeStore:
    """Lane 7's `Store` as D4 describes it: `config()` and `import_run(doc, finished=True)`."""

    def __init__(self, home, config: dict | None = None, reject: set | None = None):
        self.home = home
        self.cfg = FakeConfig(config)
        self.runs: dict[str, dict] = {}
        self.imports = 0
        self.reject = set(reject or ())

    def config(self):
        return self.cfg

    def import_run(self, doc, *, finished=True):
        assert finished is True
        self.imports += 1
        if doc["run"]["task"]["type"] in self.reject:
            raise ValueError("E120 run.task.type rejected by the fake store")
        self.runs[doc["run"]["id"]] = json.loads(json.dumps(doc))
        return doc["run"]["id"]
