"""Build the anonymized logmatch fixtures from real sessions on this machine.

Run by hand, never by pytest (the name does not start with `test_`):

    PYTHONPATH=src python tests/logmatch/make_fixtures.py \
        --claude SESSION_ID --codex THREAD_ID [--codex THREAD_ID ...]

Each named session is matched with its children on this machine, then
rewritten with only the keys the parsers read for matching and pricing:
record types, ids, timestamps, working folders, models, efforts and token
counts. Every id becomes a fake one, every working folder `/work/repo`,
every timestamp moves so each session starts at a fixed hour on 2026-09-20
(children keep their offsets from the parent), and every text field is
dropped. The builder then matches the fixtures and checks that each part's
model and four token streams equal the real session's, and writes
`fixtures/expected.json`. Real ids are never written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loopmath.ingest.base import iter_jsonl, parse_ts, ts_epoch
from loopmath.logmatch.match import LogRoots, default_roots, explain_match

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures"
CWD = "/work/repo"
BASE = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone.utc)
USAGE_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)
CODEX_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


class Fakes:
    """Stable fake ids, one sequence per kind."""

    def __init__(self) -> None:
        self.seen: dict[tuple[str, str], str] = {}

    def get(self, kind: str, real: str) -> str:
        key = (kind, real)
        if key not in self.seen:
            n = sum(1 for k in self.seen if k[0] == kind) + 1
            self.seen[key] = {
                "cc": f"00000000-0000-4000-8000-{n:012d}",
                "cx": f"019f0000-0000-7000-8000-{n:012d}",
                "agent": f"a{n:016x}",
                "req": f"req_{n:06d}",
                "turn": f"turn-{n}",
            }[kind]
        return self.seen[key]


def _shift(value, delta: float) -> str | None:
    norm = parse_ts(value)
    if norm is None:
        return None
    t = datetime.fromtimestamp(ts_epoch(norm) + delta, timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _write(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")


def _claude_line(obj: dict, fakes: Fakes, sid: str, delta: float) -> dict | None:
    if obj.get("type") != "assistant":
        return None
    msg = obj.get("message") or {}
    usage = msg.get("usage")
    out: dict = {"type": "assistant", "sessionId": sid, "cwd": CWD}
    ts = _shift(obj.get("timestamp"), delta)
    if ts:
        out["timestamp"] = ts
    if obj.get("requestId"):
        out["requestId"] = fakes.get("req", obj["requestId"])
    if obj.get("isSidechain") is True:
        out["isSidechain"] = True
        if obj.get("agentId"):
            out["agentId"] = fakes.get("agent", obj["agentId"])
    if obj.get("effort") is not None:
        out["effort"] = str(obj["effort"])
    new_msg: dict = {"content": []}
    if isinstance(msg.get("model"), str):
        new_msg["model"] = msg["model"]
    if isinstance(usage, dict):
        new_msg["usage"] = {k: int(usage.get(k) or 0) for k in USAGE_KEYS}
    if msg.get("stop_reason") in ("end_turn", "stop_sequence", "max_tokens", "tool_use"):
        new_msg["stop_reason"] = msg["stop_reason"]
    out["message"] = new_msg
    return out


def _codex_line(obj: dict, fakes: Fakes, delta: float) -> dict | None:
    kind = obj.get("type")
    payload = obj.get("payload") or {}
    ts = _shift(obj.get("timestamp"), delta)
    if kind == "session_meta":
        meta = {
            "id": fakes.get("cx", payload["id"]),
            "session_id": fakes.get("cx", payload.get("session_id") or payload["id"]),
            "timestamp": _shift(payload.get("timestamp"), delta),
            "cwd": CWD,
            "originator": payload.get("originator"),
            "cli_version": payload.get("cli_version"),
        }
        src = payload.get("source")
        if isinstance(src, dict) and "subagent" in src:
            spawn = (src.get("subagent") or {}).get("thread_spawn") or {}
            meta["source"] = {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": fakes.get("cx", spawn.get("parent_thread_id") or ""),
                        "depth": spawn.get("depth"),
                    }
                }
            }
        elif isinstance(src, str):
            meta["source"] = src
        if payload.get("parent_thread_id"):
            meta["parent_thread_id"] = fakes.get("cx", payload["parent_thread_id"])
        if payload.get("forked_from_id"):
            meta["forked_from_id"] = fakes.get("cx", payload["forked_from_id"])
        return {"timestamp": ts, "type": kind, "payload": meta}
    if kind == "turn_context":
        return {
            "timestamp": ts,
            "type": kind,
            "payload": {
                "turn_id": fakes.get("turn", str(payload.get("turn_id"))),
                "cwd": CWD,
                "model": payload.get("model"),
                "effort": payload.get("effort"),
            },
        }
    if kind == "event_msg" and payload.get("type") == "token_count":
        info = payload.get("info") or {}
        new_info = {}
        for key in ("total_token_usage", "last_token_usage"):
            usage = info.get(key)
            if isinstance(usage, dict):
                new_info[key] = {k: int(usage[k]) for k in CODEX_USAGE_KEYS if k in usage}
        return {"timestamp": ts, "type": kind, "payload": {"type": "token_count", "info": new_info or None}}
    return None


def _first_epoch(path: Path) -> float:
    for obj in iter_jsonl(path):
        norm = parse_ts(obj.get("timestamp"))
        if norm is not None:
            return ts_epoch(norm)
    raise SystemExit(f"no timestamp in {path}")


def build(claude_ids: list[str], codex_ids: list[str]) -> dict:
    real = default_roots()
    if OUT.exists():
        shutil.rmtree(OUT)
    fakes = Fakes()
    written: list[tuple[str, str, object]] = []
    hour = 0
    for harness, ids in (("claude-code", claude_ids), ("codex", codex_ids)):
        for real_id in ids:
            match, why = explain_match({"session": real_id, "harness": harness}, roots=real)
            if match is None:
                raise SystemExit(f"{harness} session not matched: {why}")
            delta = (BASE + timedelta(hours=hour)).timestamp() - _first_epoch(match.session_path)
            hour += 1
            paths = []
            for part in match.parts:
                if part["path"] not in paths:
                    paths.append(part["path"])
            if harness == "claude-code":
                sid = fakes.get("cc", real_id)
                slug = CWD.replace("/", "-")
                for p in paths:
                    path = Path(p)
                    lines = [x for x in (_claude_line(o, fakes, sid, delta) for o in iter_jsonl(path)) if x]
                    if path == match.session_path:
                        dest = OUT / "claude" / "projects" / slug / f"{sid}.jsonl"
                    else:
                        agent = fakes.get("agent", path.stem[len("agent-"):])
                        dest = OUT / "claude" / "projects" / slug / sid / "subagents" / f"agent-{agent}.jsonl"
                    _write(dest, lines)
            else:
                for p in paths:
                    path = Path(p)
                    lines = [x for x in (_codex_line(o, fakes, delta) for o in iter_jsonl(path)) if x]
                    meta = lines[0]["payload"]
                    start = datetime.fromisoformat(meta["timestamp"].replace("Z", "+00:00"))
                    name = f"rollout-{start.strftime('%Y-%m-%dT%H-%M-%S')}-{meta['id']}.jsonl"
                    _write(OUT / "codex" / "sessions" / start.strftime("%Y/%m/%d") / name, lines)
            written.append((harness, real_id, match))

    fixture_roots = LogRoots(claude=[OUT / "claude" / "projects"], codex=[OUT / "codex" / "sessions"])
    expected = {}
    for harness, real_id, real_match in written:
        fake_id = fakes.get("cc" if harness == "claude-code" else "cx", real_id)
        match, why = explain_match({"session": fake_id, "harness": harness}, roots=fixture_roots)
        if match is None:
            raise SystemExit(f"fixture {fake_id} not matched: {why}")
        got = sorted((p["role"], p["model"], tuple(p["tokens"].as_dict().values())) for p in match.parts)
        want = sorted((p["role"], p["model"], tuple(p["tokens"].as_dict().values())) for p in real_match.parts)
        if got != want:
            raise SystemExit(f"fixture {fake_id} does not reproduce the real session's parts")
        expected[fake_id] = {
            "harness": harness,
            "models": match.models,
            "children": len(match.children),
            "tokens": match.tokens.as_dict(),
            "started_at": match.started_at,
            "ended_at": match.ended_at,
            "effort": match.effort,
        }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return expected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--claude", action="append", default=[])
    ap.add_argument("--codex", action="append", default=[])
    args = ap.parse_args(argv)
    expected = build(args.claude, args.codex)
    for fake_id, row in expected.items():
        print(row["harness"], fake_id, row["models"], "children", row["children"], row["tokens"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
