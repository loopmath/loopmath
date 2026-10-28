"""How a session's end becomes a `SessionEnd`, and what the tune does with each kind of end.

    kind          policy    meaning
    usage_limit   pause     the account's own limit refused the session; every later one would be refused too
    auth          pause     the account's credentials or standing; the same
    transient     retry     the service or the network: once more after TRANSIENT_WAIT_S in the same workspace,
                            then a second transient error fails the tune
    agent_limit   outcome   the session's own limit (its turns, its time, its output, its budget flag): a failed
                            session, scored as it stands
    other, ""     outcome   anything else, and a clean end

`Host.session` (the engine) applies the policy (`contracts.ERROR_POLICY`); this module decides the kind. The
kind comes from the stream's error channels (`stream.classify`), then from how the process ended: past its
`timeout_s` is agent_limit, killed by a stop is the stop's, a crash with no result line is other.

`run_one` runs one session to its end outside the engine with a readable view on stdout, for a runner whose
sessions live in their own terminal panes; `write_request` and `read_request` carry the session's request there
as JSON (the prompt included, never the token).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from .contracts import (ERROR_KINDS, ERROR_POLICY, Limits, SessionEnd, SessionRequest, Usage, Workspace, now_iso,
                        record)
from .stream import StreamResult, parse_lines, render

__all__ = ["end_of", "policy", "status_line", "write_exit", "write_request", "read_request", "run_one"]


def end_of(label: str, role: str, got: StreamResult, *, exit: int | None, wall_s: float, timed_out: bool = False,
           killed: bool = False, timeout_s: float | None = None, model: str = "") -> SessionEnd:
    """The `SessionEnd` of a process that exited (`exit`, None when a signal ended it) with stream `got`. `model`
    is the session's own: a session that ended on an API error with no tokens and no model in its stream is one
    call on it with zero tokens."""
    kind, why = got.error_kind, got.error
    if timed_out:
        kind, why = "agent_limit", f"the session passed its timeout_s ({timeout_s:g} s) and was killed" \
            if timeout_s is not None else "the session passed its timeout_s and was killed"
    elif killed:
        kind, why = kind or "", why or "killed by a stop"
    elif not got.has_result and not kind:
        kind = "other"
        why = (f"the session ended without a result line (exit {exit})" if exit is not None
               else "the session ended on a signal without a result line")
        if got.bad_lines:
            why += f"; {got.bad_lines} stream lines did not parse"
    if kind not in ERROR_KINDS and kind:
        kind = "other"
    is_error = bool(got.is_error or timed_out or kind)
    usage = list(got.usage)
    if not usage and got.api_error and model:
        usage = [Usage(model=model, cli_cost_usd=got.cli_cost_usd)]
    return SessionEnd(label=label, role=role, exit=exit, is_error=is_error, wall_s=round(float(wall_s), 3),
                      usage=usage, text=got.text, error_kind=kind, error=why[:1000],
                      cli_cost_usd=got.cli_cost_usd, killed=killed)


def policy(end: SessionEnd) -> str:
    """pause, retry or outcome, as `ERROR_POLICY` says for the end's kind."""
    return ERROR_POLICY.get(end.error_kind or "", "outcome")


def status_line(end: SessionEnd) -> str:
    """One plain line for `status`: "paused: usage limit", "retry: transient API error", "failed: ..." or "ok"."""
    p = policy(end)
    kind = (end.error_kind or "").replace("_", " ")
    if p == "pause":
        return f"paused: {kind}"
    if p == "retry":
        return "retry: transient API error"
    if end.ok:
        return "ok"
    return f"failed: {end.error or kind or f'exit {end.exit}'}"[:200]


def write_exit(run_dir: Path, end: SessionEnd, **extra: Any) -> Path:
    """`runs/<label>/exit.json`, atomically: the ground truth that a session ended."""
    from ...store.lock import atomic_write_json

    data = {"label": end.label, "role": end.role, "exit": end.exit, "is_error": end.is_error,
            "error_kind": end.error_kind, "error": end.error, "wall_s": end.wall_s, "killed": end.killed,
            "cli_cost_usd": end.cli_cost_usd, "models": [u.model for u in end.usage], "ended": now_iso(), **extra}
    return atomic_write_json(Path(run_dir) / "exit.json", data)


def write_request(req: SessionRequest, path: Path) -> Path:
    from ...store.lock import atomic_write_json

    return atomic_write_json(path, record(req))


def read_request(path: Path) -> SessionRequest:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    w = dict(d["workspace"])
    for k in ("folder", "cfg_dir", "run_dir"):
        w[k] = Path(w[k])
    w["tmp_dir"] = Path(w["tmp_dir"]) if w.get("tmp_dir") else None
    lim = dict(d["limits"])
    lim["tools"] = tuple(lim.get("tools") or ())
    return SessionRequest(label=d["label"], role=d["role"], stage=d["stage"], prompt=d["prompt"],
                          workspace=Workspace(**w), limits=Limits(**lim), attempt=int(d.get("attempt") or 1),
                          meta=dict(d.get("meta") or {}))


def run_one(executor: Any, req: SessionRequest, *, out: TextIO | None = None, poll_s: float = 0.5) -> SessionEnd:
    """Start one session on `executor`, print a readable view of its stream to `out` while it runs, wait for its
    end (the executor writes exit.json) and return it."""
    out = sys.stdout if out is None else out
    sess = executor.start(req)
    stream = Path(req.workspace.run_dir) / "stream.jsonl"
    print(f"{req.label}: {req.role} on {req.limits.model}, pid {sess.pid}", file=out, flush=True)
    box: dict[str, Any] = {}

    def waiter() -> None:
        try:
            box["end"] = executor.wait(sess)
        except BaseException as exc:  # handed to the caller below
            box["exc"] = exc

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    pos = 0
    tail = ""

    def pump() -> None:
        nonlocal pos, tail
        try:
            with open(stream, encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except OSError:
            return
        tail += chunk
        lines = tail.split("\n")
        tail = lines.pop()
        events, _, _ = parse_lines(lines)
        for e in events:
            line = render(e)
            if line:
                print(line, file=out, flush=True)

    while t.is_alive():
        pump()
        time.sleep(poll_s)
    pump()
    if "exc" in box:
        raise box["exc"]
    end: SessionEnd = box["end"]
    print(f"{req.label}: {status_line(end)} in {end.wall_s:.0f} s", file=out, flush=True)
    return end
