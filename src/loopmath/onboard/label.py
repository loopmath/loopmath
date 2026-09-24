"""One batch labelling call through the labeller the user chose.

Owner: lane 03. Spec: design/0.1/02-commands.md sections 1 and 4, 08-lanes.md section 3;
decisions D6, D11 and D39 in the lane questions log.

The user picks the labeller; loopmath never picks a model for them (D39):

    claude:<model>    runs the user's `claude -p --model <model>`
    codex:<model>     runs the user's `codex exec --model <model>`
    command:<cmd>     runs `<cmd>` in /bin/sh in the current folder: the chunk as JSON on
                      stdin (see `command_request`), `{"labels": [...]}` on stdout; for local models
    none              no model: a keyword guess, low confidence

The batch is one approval: its expected cost is shown once and one `--yes` covers
it. It is sent in chunks of at most `CHUNK` groups, because a few months of history
does not fit one call. A chunk that fails is retried once; after that its groups are
reported as unclassified. loopmath itself opens no network connection: the user's
own CLI or command does, as a subprocess. `claude` and `codex` run with no tools, no
MCP servers and no saved session, so the labelling call does not join the history
it labels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import taskmodel

CHUNK = 150
KINDS = ("claude", "codex", "command")
FORMS = ("claude:<model> (runs your claude CLI)", "codex:<model> (runs your codex CLI)",
         "command:<cmd> (any command, such as a local model: the batch JSON on stdin, the labels JSON on stdout)",
         "none (no model; a keyword guess)")
REQUEST_SCHEMA = "loopmath.label-request/1"
CODEX_EFFORT = "low"
CHARS_PER_TOKEN = 4.0          # estimate only; the real count comes back from the CLI
OUT_TOKENS_PER_GROUP = 60      # one label is about 50 to 70 tokens of JSON
# Tokens the CLI adds to every call on top of our prompt. codex keeps its base
# instructions: measured 2026-09-23 (codex-cli 0.155.1, gpt-6-luna), one call with a
# 9,143-character stdin (about 2,300 tokens) reported 18,726 input tokens, so about
# 16,400 of overhead. claude runs with our own --system-prompt; its figure is assumed.
CALL_OVERHEAD_TOKENS = {"claude": 1500, "codex": 16500}
TIMEOUT_S = 900


class LabelError(Exception):
    """The labeller cannot run (bad spec, CLI missing)."""


@dataclass
class Labeler:
    name: str                  # "claude" | "codex" | "command" | "none"
    model: str | None = None   # claude and codex
    command: str | None = None  # command
    executable: str | None = None

    @property
    def spec(self) -> str:
        """The form the user typed, as saved in config `onboard.labeler`."""
        if self.name == "none":
            return "none"
        return f"{self.name}:{self.command if self.name == 'command' else self.model}"

    @property
    def title(self) -> str:
        """For people: `codex (gpt-6-luna)`, `your command`, `none`."""
        if self.name == "command":
            return "your command"
        return f"{self.name} ({self.model})" if self.model else self.name


def parse_labeler(spec: str, *, which: Callable[[str], str | None] = shutil.which) -> Labeler:
    """A `--labeler` or config `onboard.labeler` value to a `Labeler` (D39 forms only)."""
    from ..cli_registry import labeler_spec

    try:
        spec = labeler_spec(str(spec).strip())
    except argparse.ArgumentTypeError as exc:
        raise LabelError(f"labeller: {exc}") from exc
    if spec == "none":
        return Labeler("none")
    kind, _, rest = spec.partition(":")
    rest = rest.strip()
    if kind == "command":
        return Labeler("command", command=rest)
    exe = which(kind)
    if not exe:
        raise LabelError(f"{kind} is not on PATH; install it, or choose another labeller: " + "; ".join(FORMS))
    return Labeler(kind, model=rest, executable=exe)


def choose_labeler(requested: str | None, configured: str | None, *,
                   which: Callable[[str], str | None] = shutil.which) -> tuple[Labeler | None, str | None]:
    """(labeller, where it came from: "flag" | "config"), or (None, None) when the user
    has not chosen one. There is no default (D39)."""
    if requested:
        return parse_labeler(requested, which=which), "flag"
    if configured:
        return parse_labeler(configured, which=which), "config"
    return None, None


def _price_table(table: Any = None) -> Any:
    if table is not None:
        return table
    from ..price import load_prices

    return load_prices()


def chunks(items: list[Any], size: int = CHUNK) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_prompt(batch: list[dict]) -> str:
    """The user message for one chunk: one JSON object per line, one line per group."""
    lines = [f"Label these {len(batch)} items. Items, one JSON object per line:"]
    lines += [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in batch]
    return "\n".join(lines) + "\n"


def estimate(prompts: list[str], system: str, n_groups: int, labeler: Labeler, *, table: Any = None) -> dict:
    """Expected tokens and dollars for the whole batch, before any call is made.
    `usd` is None when the model has no price row, and always for `command:`."""
    from ..price import price_run

    calls = len(prompts)
    overhead = CALL_OVERHEAD_TOKENS.get(labeler.name, 0)
    input_tokens = sum(math.ceil((len(p) + len(system)) / CHARS_PER_TOKEN) + overhead for p in prompts)
    output_tokens = OUT_TOKENS_PER_GROUP * n_groups
    out = {"labeler": labeler.spec, "model": labeler.model, "calls": calls, "groups": n_groups,
           "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
           "usd": None, "price_todo": False, "usd_unknown": None, "basis": "estimate: characters / 4, about "
           f"{OUT_TOKENS_PER_GROUP} output tokens per group, {overhead} assumed CLI tokens per call"}
    if labeler.name == "command":
        out["usd_unknown"] = "a command reports no price"
    elif labeler.model:
        priced = price_run({"model": labeler.model,
                            "tokens": {"in": input_tokens, "cache_read": 0, "cache_write": 0, "out": output_tokens}},
                           _price_table(table))
        if priced.get("usd") is not None:
            out["usd"] = round(float(priced["usd"]), 4)
        else:
            out["usd_unknown"] = f"no price row for {labeler.model}"
        out["price_todo"] = bool(priced.get("todo"))
    return out


def command_request(batch: list[dict], *, system: str, schema: dict) -> str:
    """What a `command:` labeller reads on stdin for one chunk."""
    return json.dumps({"schema": REQUEST_SCHEMA, "instructions": system, "answer_schema": schema, "items": batch},
                      ensure_ascii=False) + "\n"


def labeler_argv(labeler: Labeler, *, system: str, schema: dict, workdir: Path) -> list[str]:
    """The exact command for one chunk. The chunk's items go on stdin."""
    if labeler.name == "claude":
        return [labeler.executable or "claude", "-p", "--model", str(labeler.model),
                "--output-format", "json", "--json-schema", json.dumps(schema, separators=(",", ":")),
                "--system-prompt", system, "--tools", "", "--strict-mcp-config",
                "--disable-slash-commands", "--no-session-persistence"]
    if labeler.name == "codex":
        schema_path = workdir / "schema.json"
        return [labeler.executable or "codex", "exec", "--model", str(labeler.model),
                "-c", f'model_reasoning_effort="{CODEX_EFFORT}"', "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--cd", str(workdir),
                "--output-schema", str(schema_path), "-o", str(workdir / "answer.txt"), "--json", "-"]
    if labeler.name == "command":
        return ["/bin/sh", "-c", str(labeler.command)]
    raise LabelError(f"no command for labeller {labeler.name!r}")


@dataclass
class LabelRun:
    labels: dict[str, dict] = field(default_factory=dict)       # group id -> validated label
    rejected: dict[str, str] = field(default_factory=dict)      # group id -> reason (unknown, invalid, missing)
    failed_chunks: list[dict] = field(default_factory=list)     # {chunk, groups, error}
    calls: int = 0
    cost: dict = field(default_factory=lambda: {"usd": 0.0, "tokens": 0, "basis": "reported by the CLI"})
    problems: list[str] = field(default_factory=list)


def _find_json(text: str) -> dict | None:
    from ..graph.labeler_parse import _find_json_object

    return _find_json_object(text or "")


def _claude_answer(stdout: str) -> tuple[dict | None, dict]:
    """(answer object, {usd, tokens}) from `claude -p --output-format json`."""
    try:
        obj = json.loads(stdout)
    except ValueError:
        return _find_json(stdout), {}
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict) and o.get("type") == "result"), {})
    if not isinstance(obj, dict):
        return None, {}
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    tokens = sum(int(usage.get(k) or 0) for k in ("input_tokens", "output_tokens",
                                                 "cache_read_input_tokens", "cache_creation_input_tokens"))
    cost = {"usd": obj.get("total_cost_usd"), "tokens": tokens}
    if obj.get("is_error"):
        return None, cost
    answer = obj.get("structured_output")
    if isinstance(answer, dict) and "labels" in answer:
        return answer, cost
    return _find_json(str(obj.get("result") or "")), cost


def _codex_answer(stdout: str, answer_path: Path, model: str | None, table: Any) -> tuple[dict | None, dict]:
    """(answer object, {usd, tokens}) from `codex exec --json -o FILE`."""
    usage: dict = {}
    for line in (stdout or "").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
            for k, v in ev["usage"].items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
    cost: dict = {}
    if usage:
        cached = int(usage.get("cached_input_tokens") or 0)
        tokens = {"in": max(0, int(usage.get("input_tokens") or 0) - cached), "cache_read": cached,
                  "cache_write": 0, "out": int(usage.get("output_tokens") or 0)}
        from ..price import price_run

        priced = price_run({"model": model, "tokens": tokens}, _price_table(table))
        cost = {"usd": priced.get("usd"), "tokens": sum(tokens.values())}
    try:
        text = answer_path.read_text(encoding="utf-8")
    except OSError:
        return None, cost
    return _find_json(text), cost


def _command_answer(stdout: str) -> tuple[dict | None, dict]:
    """(answer object, {usd, tokens}) from a `command:` labeller; cost only if it reports one."""
    answer = _find_json(stdout or "")
    reported = answer.get("cost") if isinstance(answer, dict) else None
    cost: dict = {"usd": None, "tokens": 0}
    if isinstance(reported, dict):
        usd, tokens = reported.get("usd"), reported.get("tokens")
        cost = {"usd": float(usd) if isinstance(usd, (int, float)) else None,
                "tokens": int(tokens) if isinstance(tokens, int) else 0}
    return answer, cost


def parse_answer(answer: dict | None, expected: list[str], subtypes: list[str] | None) -> tuple[dict, dict, list[str]]:
    """(labels, rejected, problems) for one chunk's answer object."""
    labels: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    problems: list[str] = []
    wanted = set(expected)
    items = answer.get("labels") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, {gid: "labeller answer unreadable" for gid in expected}, ["answer has no labels list"]
    for item in items:
        gid = item.get("id") if isinstance(item, dict) else None
        if not isinstance(gid, str) or gid not in wanted:  # a list or dict id must not crash a paid batch
            problems.append(f"answer names an id that was not asked: {str(gid)[:40]!r}")
            continue
        if gid in labels or gid in rejected:
            problems.append(f"answer repeats id {gid}")
            continue
        label, notes = taskmodel.validate_label(item, subtypes)
        if label is None:
            rejected[gid] = "labeller said unknown" if notes == ["type unknown"] else "labeller answer invalid: " + "; ".join(notes)
        else:
            labels[gid] = label
            problems += [f"{gid}: {n}" for n in notes]
    for gid in expected:
        if gid not in labels and gid not in rejected:
            rejected[gid] = "labeller skipped it"
    return labels, rejected, problems


def chunk_stdin(labeler: Labeler, batch: list[dict], *, system: str, schema: dict) -> str:
    """Exactly what one chunk's call reads on stdin. `claude` gets the instructions as
    `--system-prompt`; `codex` reads them at the top of the prompt."""
    if labeler.name == "command":
        return command_request(batch, system=system, schema=schema)
    if labeler.name == "codex":
        return system + "\n\n" + build_prompt(batch)
    return build_prompt(batch)


def run_batches(batches: list[list[dict]], labeler: Labeler, *, subtypes: list[str] | None = None,
                runner: Callable[..., Any] = subprocess.run, table: Any = None, timeout: float = TIMEOUT_S,
                progress: Callable[[int, int], None] | None = None, attempts: int = 2,
                trace: list[dict] | None = None) -> LabelRun:
    """Send every chunk (after the caller has the user's yes). `attempts` 2 is one retry
    per chunk (D6). When `trace` is a list, each call's argv, stdin, exit code and
    output are appended to it, so a caller can show exactly what was sent."""
    system = taskmodel.label_instructions(subtypes)
    schema = taskmodel.label_schema(subtypes)
    result = LabelRun()
    if labeler.name == "command":
        result.cost["basis"] = "reported by your command"
    usd_known = True
    with tempfile.TemporaryDirectory(prefix="loopmath-label-") as tmp:
        workdir = Path(tmp)
        (workdir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        argv = labeler_argv(labeler, system=system, schema=schema, workdir=workdir)
        for index, batch in enumerate(batches):
            if progress:
                progress(index + 1, len(batches))
            expected = [item["id"] for item in batch]
            stdin = chunk_stdin(labeler, batch, system=system, schema=schema)
            error = "no attempt"
            answer = None
            for _attempt in range(max(1, attempts)):
                result.calls += 1
                answer_path = workdir / "answer.txt"
                answer_path.unlink(missing_ok=True)
                call: dict = {"chunk": index + 1, "argv": list(argv), "stdin": stdin}
                if trace is not None:
                    trace.append(call)
                try:
                    # claude and codex run in the empty temp folder; a command runs where the user is,
                    # so relative paths in it work.
                    proc = runner(argv, input=stdin, capture_output=True, text=True, timeout=timeout,
                                  cwd=None if labeler.name == "command" else str(workdir))
                except subprocess.TimeoutExpired:
                    error = f"timed out after {int(timeout)} s"
                    usd_known = False
                    call["error"] = error
                    continue
                except OSError as exc:
                    error = f"could not start {labeler.name}: {exc.strerror or exc}"
                    call["error"] = error
                    continue
                call.update({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
                if labeler.name == "claude":
                    answer, cost = _claude_answer(proc.stdout or "")
                elif labeler.name == "codex":
                    answer, cost = _codex_answer(proc.stdout or "", answer_path, labeler.model, table)
                    if answer_path.exists():
                        call["answer_file"] = answer_path.read_text(encoding="utf-8", errors="replace")
                else:
                    answer, cost = _command_answer(proc.stdout or "")
                if cost.get("usd") is None:
                    usd_known = False
                result.cost["usd"] += float(cost.get("usd") or 0.0)
                result.cost["tokens"] += int(cost.get("tokens") or 0)
                if proc.returncode != 0:
                    error = f"{labeler.name} exited {proc.returncode}"
                    answer = None
                    continue
                if answer is None:
                    error = "answer was not JSON with a labels list"
                    continue
                break
            if answer is None:
                result.failed_chunks.append({"chunk": index + 1, "groups": len(batch), "error": error})
                for gid in expected:
                    result.rejected[gid] = f"labeller failed ({error})"
                continue
            labels, rejected, problems = parse_answer(answer, expected, subtypes)
            result.labels.update(labels)
            result.rejected.update(rejected)
            result.problems += problems
    result.cost["usd"] = round(result.cost["usd"], 4) if usd_known else None
    return result


def keyword_labels(summaries: list[dict], prompts: dict[str, str | None]) -> tuple[dict, dict]:
    """`--labeler none`: (labels, rejected) from `taskmodel.guess_type` on each first prompt."""
    labels: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    for item in summaries:
        gid = item["id"]
        text = prompts.get(gid)
        if not text:
            rejected[gid] = "no prompt in the session"
            continue
        kind, confidence = taskmodel.guess_type(text)
        if kind is None:
            rejected[gid] = "no keyword matched"
            continue
        labels[gid] = {"type": kind, "subtype": None, "features": {}, "confidence": confidence, "title": ""}
    return labels, rejected
