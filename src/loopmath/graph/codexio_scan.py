"""Codex rollout scanner implementation split from :mod:`loopmath.graph.codexio`."""
from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter
from pathlib import Path

from . import bashwrites
from .scan import iter_jsonl

from .codexio_parse import *

def _command_string(value) -> str | None:
    """A shell command from the argument value: a string as is; a list either
    unwrapped from `[shell, -c|-lc, cmd]` or joined with shell quoting."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        if len(value) >= 3 and value[0] in _SHELL_WRAPPERS and value[1] in ("-c", "-lc", "-ic", "-lic"):
            return value[-1]
        if value and value[0] == "apply_patch" and len(value) == 2:
            return "apply_patch <<'EOF'\n" + value[1] + "\nEOF"
        return shlex.join(value)
    return None


def _shell_args(payload: dict) -> tuple[list[dict], int]:
    """`([{cmd, workdir}, ...], n_unparsed)` for one shell-like tool call payload."""
    kind, name = payload.get("type"), payload.get("name")
    if kind == "local_shell_call":
        action = payload.get("action") or {}
        cmd = _command_string(action.get("command"))
        if cmd is None:
            return [], 1
        return [{"cmd": cmd, "workdir": action.get("working_directory")}], 0
    if kind == "custom_tool_call" and name == "exec":
        script = payload.get("input")
        if not isinstance(script, str):
            return [], 1
        calls, unparsed = exec_commands_from_js(script)
        out = []
        for c in calls:
            cmd = _command_string(c.get("cmd", c.get("command")))
            if cmd is None:
                unparsed += 1
                continue
            out.append({"cmd": cmd, "workdir": c.get("workdir")})
        return out, unparsed
    if kind == "function_call" and name in _SHELL_FUNCTIONS:
        raw = payload.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            return [], 1
        if not isinstance(args, dict):
            return [], 1
        cmd = _command_string(args.get("command", args.get("cmd")))
        if cmd is None:
            return [], 1
        return [{"cmd": cmd, "workdir": args.get("workdir") or args.get("working_directory")}], 0
    return [], 0


def _patch_text(payload: dict) -> str | None:
    """The patch envelope of an apply_patch tool call, or None when absent."""
    kind = payload.get("type")
    if kind == "custom_tool_call":
        text = payload.get("input")
        return text if isinstance(text, str) else None
    raw = payload.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        for key in ("input", "patch"):
            if isinstance(args.get(key), str):
                return args[key]
    return None


def _split_patch_from_shell(cmd: str) -> tuple[str | None, str]:
    """`(patch, remainder)`: the envelope embedded in a shell `apply_patch <<'EOF' ...`
    command and the command with the envelope cut out (so the shell reads and writes
    around it can still be scanned); `(None, cmd)` when there is none."""
    if "apply_patch" not in cmd or _PATCH_BEGIN not in cmd:
        return None, cmd
    start = cmd.index(_PATCH_BEGIN)
    end = cmd.find(_PATCH_END, start)
    end = len(cmd) if end < 0 else end + len(_PATCH_END)
    return cmd[start:end], cmd[:start] + cmd[end:]


# --- tool results ----------------------------------------------------------------------


def _result_text(output) -> tuple[str, int | None]:
    """`(text, exit_code)` of a tool result payload's `output`, which codex has
    serialized as a string, as `{output, metadata: {exit_code}}`, or as a list of
    `{type: input_text, text}` parts."""
    if isinstance(output, dict):
        meta = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
        code = meta.get("exit_code")
        text = output.get("output")
        return (text if isinstance(text, str) else json.dumps(output)), (code if isinstance(code, int) else None)
    if isinstance(output, list):
        parts = [p.get("text") for p in output if isinstance(p, dict) and isinstance(p.get("text"), str)]
        text = "\n".join(parts)
        return text, _embedded_exit_code(text)
    if isinstance(output, str):
        m = _EXIT_CODE.search(output)
        return output, (int(m.group(1)) if m else None)
    return "", None


def _embedded_exit_code(text: str) -> int | None:
    """The exit code an `exec` script's joined output carries for the commands it
    wrapped (`_EMBEDDED_EXIT`); with several, a non-zero one; None when it carries
    none."""
    codes = [int(next(g for g in m.groups() if g is not None)) for m in _EMBEDDED_EXIT.finditer(text)]
    if not codes:
        return None
    return next((c for c in codes if c != 0), 0)


def _result_status(text: str, code: int | None, wrapped: bool) -> str:
    """`ok`, `fail`, `completed`, `running` or `unknown` for a tool call's result. A
    direct apply_patch call answers with the `Success.` banner or an exit code; a
    shell call with an exit code; a wrapped call's script reports `Script completed`
    (the script ran to its end; what each call in it did is known only from what the
    script printed), `Script running` (a yield) or `Script failed`. For a wrapped call
    the script's own line comes before an exit code embedded in its output: that code
    belongs to a command the script ran, not to the patch."""
    if text.startswith(_SUCCESS_BANNER):
        return "ok"
    if "apply_patch verification failed" in text or text.startswith("Script failed") or text.startswith("Script error"):
        return "fail"
    if wrapped and text.startswith("Script completed"):
        return "completed"
    if wrapped and text.startswith("Script running"):
        return "running"
    if code is not None:
        return "ok" if code == 0 else "fail"
    if not wrapped and text.startswith("Success"):
        return "ok"
    return "unknown"


def _shell_failed(result: tuple[str, int | None] | None) -> bool:
    """True when a shell call's matched result reports failure: a non-zero exit code
    in any result form (dict metadata, string, or embedded in a script's output), or
    a `Script failed` wrapper."""
    if result is None:
        return False
    text, code = result
    if code is not None:
        return code != 0
    return _result_status(text, None, wrapped=True) == "fail"


_UNCONFIRMED_HOW = {"completed": "script completed", "running": "script running", "unmatched": "no result", "ok": "result silent", "unknown": "result silent"}


def _confirmed_paths(text: str) -> list[str]:
    """Paths listed under a `Success. Updated the following files:` banner."""
    out: list[str] = []
    for m in re.finditer(re.escape(_SUCCESS_BANNER), text):
        block = text[m.end():]
        for line in block.split("\n")[1:]:
            lm = _LISTED_PATH.match(line)
            if not lm:
                break
            out.append(lm.group(1))
    return out


# --- the launching command -------------------------------------------------------------


def _shell_segments(command: str) -> list[str]:
    """The command split at `&&`, `||`, `;`, `|`, `&` and newlines outside quotes
    (`>&` and `<&` are redirections, not separators)."""
    segs: list[str] = []
    cur: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                cur.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.extend((c, command[i + 1]))
            i += 2
            continue
        if c in "\"'":
            quote = c
            cur.append(c)
        elif c == "\n" or c == ";":
            segs.append("".join(cur))
            cur = []
        elif c in "&|":
            if c == "&" and cur and cur[-1] in "<>":
                cur.append(c)
            else:
                segs.append("".join(cur))
                cur = []
                if i + 1 < n and command[i + 1] == c:
                    i += 1
        else:
            cur.append(c)
        i += 1
    segs.append("".join(cur))
    return [s for s in segs if s.strip()]


def codex_output_paths(command: str) -> list[list[str]]:
    """Per `codex exec` invocation in the command, its `-o` /
    `--output-last-message` paths (quotes removed, `$VAR` kept literal)."""
    found: list[list[str]] = []
    for seg in _shell_segments(command):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        for i, tok in enumerate(toks[:-1]):
            if Path(tok).name == "codex" and toks[i + 1] in ("exec", "e"):
                paths: list[str] = []
                j = i + 2
                while j < len(toks):
                    t = toks[j]
                    if t in ("-o", "--output-last-message") and j + 1 < len(toks):
                        paths.append(toks[j + 1])
                        j += 2
                        continue
                    if t.startswith("--output-last-message="):
                        paths.append(t.split("=", 1)[1])
                    elif t.startswith("-o") and len(t) > 2 and not t.startswith("--"):
                        paths.append(t[2:])
                    j += 1
                found.append(paths)
                break
    return found


def output_writes_from_launch(command: str, ts: str | None, cwd: str | None = None) -> list[dict]:
    """Writes for the `-o` / `--output-last-message` files of the `codex exec`
    invocation in a launching command: tier `verified`, the codex CLI writes them
    itself. With several `codex exec` invocations in one command the writer is not
    settled: tier `heuristic`, `ambiguous: True`."""
    per_call = codex_output_paths(command)
    ambiguous = len(per_call) > 1
    out = []
    for paths in per_call:
        for raw in paths:
            e = _entry(ts, raw, "heuristic" if ambiguous else "verified", "codex -o", cwd)
            if ambiguous:
                e["ambiguous"] = True
            out.append(e)
    return out


# --- the scan --------------------------------------------------------------------------


def scan_codex_session(path: Path, launch_command: str | None = None) -> dict:
    """Writes, reads and origin of one codex rollout (module docstring)."""
    writes: list[dict] = []
    reads: list[dict] = []
    artifact_facts: list[dict] = []
    # Every origin field is present from the start; a rollout without `session_meta`
    # returns them all None, all listed as missing, and counts the missing record.
    origin: dict = {k: None for k in _ORIGIN_FIELDS}
    origin["missing"] = list(_ORIGIN_FIELDS)
    origin["tier"] = "reported"
    origin["session_meta"] = False
    meta = {
        "records": 0, "shell_calls": 0, "patch_calls": 0, "unparsed_shell_calls": 0, "unparsed_patch_calls": 0,
        "patch_deletes": 0, "patch_calls_failed": 0, "patch_calls_unmatched": 0, "patch_writes_unconfirmed": 0,
        "shell_calls_failed": 0, "origin_missing_session_meta": 0, "samples": [],
    }
    bashwrite_exclusions: Counter = Counter()
    session_cwd: str | None = None
    turn_cwd: str | None = None
    last_ts: str | None = None
    calls: list[tuple[str | None, dict, str | None]] = []  # (ts, payload, cwd at the call)
    results: dict[str, tuple[str, int | None]] = {}

    for obj in iter_jsonl(path):
        meta["records"] += 1
        ts = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None
        if ts:
            last_ts = ts
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        top = obj.get("type")
        if top == "session_meta":
            if origin.get("session_meta") is False:  # the first session_meta record wins
                origin = {k: payload.get(k) if isinstance(payload.get(k), str) else None for k in _ORIGIN_FIELDS}
                origin["missing"] = [k for k in _ORIGIN_FIELDS if origin[k] is None]
                origin["tier"] = "reported"
                session_cwd = origin["cwd"]
            continue
        if top == "turn_context" and isinstance(payload.get("cwd"), str):
            turn_cwd = payload["cwd"]
            continue
        kind = payload.get("type")
        if kind in _CALL_TYPES:
            calls.append((ts, payload, turn_cwd or session_cwd))
        elif kind in _OUTPUT_TYPES and isinstance(payload.get("call_id"), str):
            results[payload["call_id"]] = _result_text(payload.get("output"))

    def note(text) -> None:
        if len(meta["samples"]) < _MAX_SAMPLES:
            meta["samples"].append(str(text)[:200])

    def add_patch(patch: str, ts: str | None, cwd: str | None, result: tuple[str, int | None] | None, wrapped: bool) -> None:
        meta["patch_calls"] += 1
        paths, deletes = patch_paths(patch)
        raw_facts = patch_artifact_facts(patch)
        move_diff_lines = sum(
            fact.get("_unmodeled_diff_lines", 0)
            for fact in raw_facts
            if isinstance(fact.get("_unmodeled_diff_lines"), int)
        )
        move_diff_bodies = sum("_unmodeled_diff_lines" in fact for fact in raw_facts)
        if move_diff_bodies:
            meta["patch_move_diff_bodies_unmodeled"] = (
                meta.get("patch_move_diff_bodies_unmodeled", 0) + move_diff_bodies
            )
            meta["patch_move_diff_lines_unmodeled"] = (
                meta.get("patch_move_diff_lines_unmodeled", 0) + move_diff_lines
            )
        meta["patch_deletes"] += deletes
        if not paths and not deletes:
            meta["unparsed_patch_calls"] += 1
            note(patch)
            return
        text, code = result if result is not None else ("", None)
        status = _result_status(text, code, wrapped) if result is not None else "unmatched"
        confirmed = {_abs(p, cwd)[0] for p in _confirmed_paths(text)}
        confirmed_for_call = any(_abs(p, cwd)[0] in confirmed for p, _ in paths)
        if status == "fail" and not confirmed_for_call:
            meta["patch_calls_failed"] += 1
            note(text)
        elif status == "unmatched":
            meta["patch_calls_unmatched"] += 1
        emitted: dict[str, tuple[str, bool]] = {}
        for p, how in paths:
            e = _entry(ts, p, "verified", how, cwd)
            if e["path"] in confirmed:
                # Mechanical confirmation: the result names the path.
                if wrapped:
                    e["how"] = how + ", result lists path"
                writes.append(e)
                emitted[e["path"]] = (str(e["tier"]), True)
            elif status == "ok" and not wrapped:
                writes.append(e)
                emitted[e["path"]] = (str(e["tier"]), True)
            elif status in _UNCONFIRMED_HOW:
                # The call is the record of an attempt; nothing in the result names the path.
                e["tier"] = "heuristic"
                e["how"] = how + ", " + _UNCONFIRMED_HOW[status]
                e["unconfirmed"] = True
                meta["patch_writes_unconfirmed"] += 1
                writes.append(e)
                emitted[e["path"]] = (str(e["tier"]), False)

        operation_paths = [_abs(p, cwd)[0] for p in (raw_facts[0].get("operation_paths", []) if raw_facts else [])]
        for raw_fact in raw_facts:
            resolved, relative = _abs(str(raw_fact["path"]), cwd)
            emitted_state = emitted.get(resolved)
            # A delete has no public write record.  Retain it only when the tool
            # result mechanically establishes that the operation ran.
            if emitted_state is None:
                observed = resolved in confirmed or (status == "ok" and not wrapped)
                if raw_fact.get("operation") != "delete" or not observed:
                    continue
                tier = "verified"
            else:
                tier, observed = emitted_state
            fact = {
                "ts": ts,
                "path": resolved,
                "operation": raw_fact["operation"],
                "operation_paths": operation_paths,
                "tier": tier,
                "observed": observed,
            }
            if relative:
                fact["relative"] = True
            for field in ("lines_added", "lines_removed"):
                if field in raw_fact:
                    fact[field] = raw_fact[field]
                    # An unconfirmed patch is only an attempted write.  Its recorded
                    # diff can be retained, but never at a stronger tier than that
                    # public write record.
                    fact[f"{field}_tier"] = tier
            artifact_facts.append(fact)

    for ts, payload, cwd_default in calls:
        name = payload.get("name")
        result = results.get(payload.get("call_id")) if isinstance(payload.get("call_id"), str) else None
        if name == "apply_patch":
            patch = _patch_text(payload)
            if patch is None:
                meta["patch_calls"] += 1
                meta["unparsed_patch_calls"] += 1
                note(payload.get("input") or payload.get("arguments"))
            else:
                add_patch(patch, ts, cwd_default, result, wrapped=False)
            continue
        if payload.get("type") == "custom_tool_call" and name == "exec" and isinstance(payload.get("input"), str):
            patches, unparsed = patches_from_js(payload["input"])
            if unparsed:
                meta["patch_calls"] += unparsed
                meta["unparsed_patch_calls"] += unparsed
                note(payload["input"])
            for patch in patches:
                add_patch(patch, ts, cwd_default, result, wrapped=True)
        cmds, unparsed = _shell_args(payload)
        if unparsed:
            meta["unparsed_shell_calls"] += unparsed
            note(payload.get("input") or payload.get("arguments") or payload.get("action"))
        failed = _shell_failed(result)
        if cmds and failed:
            meta["shell_calls_failed"] += 1
        for call in cmds:
            meta["shell_calls"] += 1
            cmd = call["cmd"]
            cwd = call.get("workdir") if isinstance(call.get("workdir"), str) else cwd_default
            patch, remainder = _split_patch_from_shell(cmd)
            if patch is not None:
                add_patch(patch, ts, cwd, result, wrapped=False)
            entries_w = bashwrites.writes_from_command(remainder, ts, cwd, exclusions=bashwrite_exclusions)
            entries_r = bashwrites.reads_from_command(remainder, ts, cwd, exclusions=bashwrite_exclusions)
            if failed:
                for e in entries_w + entries_r:
                    e["failed"] = True
                    if len(cmds) > 1:
                        # The script's output does not say which of its commands failed.
                        e["failed_scope"] = "script"
            writes.extend(entries_w)
            reads.extend(entries_r)

    if launch_command:
        writes.extend(output_writes_from_launch(launch_command, last_ts, session_cwd))
    if origin.get("session_meta") is False:
        meta["origin_missing_session_meta"] = 1
    if origin["cwd"] is None and turn_cwd:
        origin["cwd"] = turn_cwd
        origin["cwd_from"] = "turn_context"
        origin["missing"] = [k for k in origin["missing"] if k != "cwd"]
    for reason, count in sorted(bashwrite_exclusions.items()):
        meta[f"bashwrites_excluded_{reason}"] = count
    out = {"writes": writes, "reads": reads}
    if artifact_facts:
        out["artifact_facts"] = artifact_facts
    out.update({"origin": origin, "meta": meta})
    return out
