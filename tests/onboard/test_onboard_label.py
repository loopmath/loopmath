"""The batch labeller: the user's choice (D39), the cost estimate, the calls, the answers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath.onboard import label as L
from loopmath.taskmodel import FEATURES

WHICH = {"claude": "/opt/bin/claude", "codex": "/opt/bin/codex"}.get


def _labels(ids, kind="bug_fix", confidence=0.9):
    return {"labels": [{"id": i, "type": kind, "subtype": None, "confidence": confidence, "title": "A task",
                        "features": {spec.key: "unknown" for spec in FEATURES.values()} | {"size": "s"}}
                       for i in ids]}


def _ids(stdin: str) -> list[str]:
    out = []
    for line in stdin.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and str(obj.get("id", "")).startswith("grp_"):
            out.append(obj["id"])
    return out


def _items(n, start=0):
    return [{"id": f"grp_{i:03d}", "prompt": f"task {i}"} for i in range(start, start + n)]


class Runner:
    """A fake `subprocess.run`: `answers` is a list of callables (argv, stdin) -> CompletedProcess."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, argv, *, input, capture_output, text, timeout, cwd):
        assert capture_output and text and Path(cwd).is_dir()
        self.calls.append((argv, input))
        return self.answers.pop(0)(argv, input)


def claude_ok(kind="bug_fix", drop=0, extra=None):
    def answer(argv, stdin):
        ids = _ids(stdin)[drop:]
        body = _labels(ids, kind)
        if extra:
            body["labels"].append(extra)
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "type": "result", "is_error": False, "result": "", "structured_output": body,
            "total_cost_usd": 0.0125, "usage": {"input_tokens": 3000, "output_tokens": 400,
                                                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}), "")
    return answer


def claude_bad(argv, stdin):
    return subprocess.CompletedProcess(argv, 0, json.dumps({"type": "result", "is_error": False, "result": "sorry",
                                                            "total_cost_usd": 0.001, "usage": {}}), "")


def exits(code):
    return lambda argv, stdin: subprocess.CompletedProcess(argv, code, "", "boom")


def codex_ok(argv, stdin):
    out = Path(argv[argv.index("-o") + 1])
    out.write_text(json.dumps(_labels(_ids(stdin), "docs")))
    events = [{"type": "thread.started"},
              {"type": "turn.completed", "usage": {"input_tokens": 10000, "cached_input_tokens": 4000, "output_tokens": 500}}]
    return subprocess.CompletedProcess(argv, 0, "\n".join(json.dumps(e) for e in events), "")


# ---------------------------------------------------------------- the choice (D39)
def test_parse_labeler_forms():
    lab = L.parse_labeler("codex:gpt-6-luna", which=WHICH)
    assert (lab.name, lab.model, lab.executable, lab.spec, lab.title) == (
        "codex", "gpt-6-luna", "/opt/bin/codex", "codex:gpt-6-luna", "codex (gpt-6-luna)")
    assert L.parse_labeler("claude:claude-haiku-4-5", which=WHICH).model == "claude-haiku-4-5"
    cmd = L.parse_labeler("command:ollama run qwen3 --format json", which=WHICH)
    assert (cmd.name, cmd.command, cmd.model, cmd.spec) == ("command", "ollama run qwen3 --format json", None,
                                                            "command:ollama run qwen3 --format json")
    assert L.parse_labeler("none").spec == "none"


@pytest.mark.parametrize("spec", ["claude", "codex", "claude:", "command: ", "local:qwen", ""])
def test_parse_labeler_rejects_a_missing_model(spec):
    with pytest.raises(L.LabelError, match="claude:<model>"):
        L.parse_labeler(spec, which=WHICH)


def test_parse_labeler_needs_the_cli_on_path():
    with pytest.raises(L.LabelError, match="codex is not on PATH"):
        L.parse_labeler("codex:gpt-6-luna", which=lambda name: None)


def test_choose_labeler_has_no_default():
    assert L.choose_labeler(None, None, which=WHICH) == (None, None)
    lab, source = L.choose_labeler("codex:gpt-6-luna", "claude:claude-haiku-4-5", which=WHICH)
    assert (lab.spec, source) == ("codex:gpt-6-luna", "flag")
    lab, source = L.choose_labeler(None, "claude:claude-haiku-4-5", which=WHICH)
    assert (lab.spec, source) == ("claude:claude-haiku-4-5", "config")


# ---------------------------------------------------------------- estimate and commands
def test_estimate():
    prompts = [L.build_prompt(b) for b in L.chunks(_items(160))]
    haiku = L.estimate(prompts, "rules", 160, L.parse_labeler("claude:claude-haiku-4-5", which=WHICH))
    assert haiku["calls"] == 2 and haiku["groups"] == 160 and haiku["labeler"] == "claude:claude-haiku-4-5"
    assert haiku["tokens"]["output"] == 160 * L.OUT_TOKENS_PER_GROUP
    assert haiku["usd"] is not None and haiku["usd"] > 0 and haiku["usd_unknown"] is None
    luna = L.estimate(prompts, "rules", 160, L.parse_labeler("codex:gpt-6-luna", which=WHICH),
                      table=_Table({}))
    assert luna["usd"] is None and luna["usd_unknown"] == "no price row for gpt-6-luna"
    cmd = L.estimate(prompts, "rules", 160, L.parse_labeler("command:x"))
    assert cmd["usd"] is None and cmd["usd_unknown"] == "a command reports no price"


class _Table:
    def __init__(self, rates):
        self.rates = rates

    def rate(self, model):
        return self.rates.get(model)

    def is_todo(self, model):
        return False


def test_argv_and_stdin_per_labeller(tmp_path):
    schema, system = {"type": "object"}, "RULES"
    batch = _items(2)
    claude = L.parse_labeler("claude:claude-haiku-4-5", which=WHICH)
    argv = L.labeler_argv(claude, system=system, schema=schema, workdir=tmp_path)
    assert argv[:4] == ["/opt/bin/claude", "-p", "--model", "claude-haiku-4-5"]
    assert argv[argv.index("--system-prompt") + 1] == "RULES" and argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv and "--bare" not in argv
    assert "RULES" not in L.chunk_stdin(claude, batch, system=system, schema=schema)
    codex = L.parse_labeler("codex:gpt-6-luna", which=WHICH)
    argv = L.labeler_argv(codex, system=system, schema=schema, workdir=tmp_path)
    assert argv[:5] == ["/opt/bin/codex", "exec", "--model", "gpt-6-luna", "-c"]
    assert {"--ephemeral", "--skip-git-repo-check", "--json"} <= set(argv) and argv[-1] == "-"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    stdin = L.chunk_stdin(codex, batch, system=system, schema=schema)
    assert stdin.startswith("RULES\n\n") and _ids(stdin) == ["grp_000", "grp_001"]
    cmd = L.parse_labeler("command:my-labeller --fast")
    assert L.labeler_argv(cmd, system=system, schema=schema, workdir=tmp_path) == ["/bin/sh", "-c", "my-labeller --fast"]
    request = json.loads(L.chunk_stdin(cmd, batch, system=system, schema=schema))
    assert request == {"schema": L.REQUEST_SCHEMA, "instructions": "RULES", "answer_schema": schema, "items": batch}


# ---------------------------------------------------------------- calls and answers
def test_claude_batch_in_chunks():
    runner = Runner(claude_ok(), claude_ok())
    trace: list = []
    run = L.run_batches(L.chunks(_items(151)), L.parse_labeler("claude:claude-haiku-4-5", which=WHICH),
                        runner=runner, trace=trace)
    assert len(runner.calls) == 2 and run.calls == 2 and len(run.labels) == 151 and run.rejected == {}
    assert run.cost["usd"] == 0.025 and run.cost["tokens"] == 6800
    assert [len(_ids(stdin)) for _, stdin in runner.calls] == [150, 1]
    assert [t["stdin"] for t in trace] == [stdin for _, stdin in runner.calls] and trace[0]["returncode"] == 0
    assert run.labels["grp_000"] == {"type": "bug_fix", "subtype": None, "features": {"size": "s"},
                                     "confidence": 0.9, "title": "A task"}


def test_partial_unknown_and_stray_answers():
    stray = _labels(["grp_zzz"])["labels"][0]
    runner = Runner(claude_ok(drop=1, extra=stray))
    run = L.run_batches([_items(3)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH), runner=runner)
    assert set(run.labels) == {"grp_001", "grp_002"}
    assert run.rejected == {"grp_000": "labeller skipped it"}
    assert any("not asked" in p for p in run.problems)
    unknown = L.run_batches([_items(1)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH),
                            runner=Runner(claude_ok(kind="unknown")))
    assert unknown.rejected == {"grp_000": "labeller said unknown"}


@pytest.mark.parametrize("bad", [[], {}, ["grp_1"], {"id": "grp_1"}, 7, None])
def test_an_unhashable_or_odd_id_is_a_problem_not_a_crash(bad):
    labels, rejected, problems = L.parse_answer({"labels": [{"id": bad, "type": "docs"}]}, ["grp_1"], None)
    assert labels == {} and rejected == {"grp_1": "labeller skipped it"}
    assert len(problems) == 1 and "not asked" in problems[0]


def test_one_retry_then_unclassified():
    run = L.run_batches([_items(2)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH),
                        runner=Runner(claude_bad, claude_ok()))
    assert run.calls == 2 and len(run.labels) == 2 and run.failed_chunks == []
    run = L.run_batches([_items(2), _items(1, 5)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH),
                        runner=Runner(exits(1), exits(1), claude_ok()))
    assert run.calls == 3 and run.failed_chunks == [{"chunk": 1, "groups": 2, "error": "claude exited 1"}]
    assert run.rejected == {"grp_000": "labeller failed (claude exited 1)", "grp_001": "labeller failed (claude exited 1)"}
    assert set(run.labels) == {"grp_005"}


def test_single_attempt_makes_one_call():
    runner = Runner(exits(2))
    run = L.run_batches([_items(1)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH), runner=runner, attempts=1)
    assert run.calls == 1 and len(runner.calls) == 1 and run.failed_chunks[0]["error"] == "claude exited 2"


def test_timeouts_and_missing_cli():
    def timeout(argv, stdin):
        raise subprocess.TimeoutExpired(argv, 1)

    def missing(argv, stdin):
        raise FileNotFoundError(2, "No such file or directory")

    run = L.run_batches([_items(1)], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH),
                        runner=Runner(timeout, missing), timeout=1)
    assert run.failed_chunks[0]["error"] == "could not start claude: No such file or directory"
    assert run.cost["usd"] is None  # a call that timed out may have spent money


def test_codex_answer_file_and_usage():
    table = _Table({"gpt-6-luna": {"input": 0.5, "cache_read": 0.05, "cache_write": 0.0, "output": 2.0}})
    run = L.run_batches([_items(2)], L.parse_labeler("codex:gpt-6-luna", which=WHICH), runner=Runner(codex_ok),
                        table=table)
    assert {v["type"] for v in run.labels.values()} == {"docs"}
    assert run.cost["tokens"] == 10500
    assert run.cost["usd"] == pytest.approx(round((6000 * 0.5 + 4000 * 0.05 + 500 * 2.0) / 1e6, 4))
    unpriced = L.run_batches([_items(2)], L.parse_labeler("codex:gpt-6-luna", which=WHICH), runner=Runner(codex_ok),
                             table=_Table({}))
    assert len(unpriced.labels) == 2 and unpriced.cost["usd"] is None


def test_command_labeller_end_to_end(tmp_path):
    """A real local command (no model): reads the request on stdin, answers on stdout."""
    script = tmp_path / "labeller.py"
    script.write_text(
        "import json, sys\n"
        "req = json.load(sys.stdin)\n"
        "assert req['schema'] == 'loopmath.label-request/1' and 'Do not use any tool' in req['instructions']\n"
        "labels = [{'id': it['id'], 'type': 'research', 'subtype': None, 'features': {}, 'confidence': 0.7,\n"
        "           'title': 'Look into it'} for it in req['items']]\n"
        "print('loading model...')\n"
        "print(json.dumps({'labels': labels, 'cost': {'usd': 0, 'tokens': 42}}))\n")
    lab = L.parse_labeler(f"command:{sys.executable} {script}")
    run = L.run_batches([_items(3)], lab)
    assert run.failed_chunks == [] and {v["type"] for v in run.labels.values()} == {"research"}
    assert run.cost == {"usd": 0.0, "tokens": 42, "basis": "reported by your command"}
    silent = L.run_batches([_items(1)], L.parse_labeler(f"command:{sys.executable} -c 'import sys; sys.stdin.read()'"),
                           attempts=1)
    assert silent.failed_chunks[0]["error"] == "answer was not JSON with a labels list" and silent.cost["usd"] is None


def test_keyword_labels():
    summaries = [{"id": "grp_a"}, {"id": "grp_b"}, {"id": "grp_c"}]
    labels, rejected = L.keyword_labels(summaries, {"grp_a": "Fix the crash", "grp_b": "hello", "grp_c": None})
    assert labels == {"grp_a": {"type": "bug_fix", "subtype": None, "features": {}, "confidence": 0.4, "title": ""}}
    assert rejected == {"grp_b": "no keyword matched", "grp_c": "no prompt in the session"}
