"""`loopmath guide flow`, the flow topic, against the commands it teaches: every `loopmath flow ...` line parses with
the real parser and the topic's last section names every flag; every member list it gives is what the command
emits, in order, under the schema name it gives; every refusal it quotes is the command's own words; its example
answers are the walk's, and its example row is a run of the example plan that `flow record` takes with no
finding; its prefill summary is what `flow init` prints from logged runs; what a command prints under `--json`
when it exits 1 is what the last section says. The commands run on a fresh flow folder and store; the background
refit is not started."""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.cli_registry import FLOW_SAVING, FLOW_VERBS
from loopmath.flows import load, walk
from loopmath.flows.check import STATE_KEYS
from loopmath.flows.commands import SHOW_KEYS, WRITER_PLAN_KEYS
from loopmath.flows.judges import CALIBRATED_AT, MAX_SCORED_JUDGES
from loopmath.flows.plan import FITTED_PAIRS
from loopmath.flows.records import EXT_PREFILL
from loopmath.flows.status import STATUS_KEYS, TAILS
from loopmath.harness_runs import to_ocp
from loopmath.skill import guide
from loopmath.store import fitjob
from tests.flows.test_flow_prefill import INFERRED_ANSWERS, ROWS, _plain, _stamped

TEXT = guide.text("flow")
FLAT = " ".join(TEXT.split())  # a wrapped line reads as one
SH = re.compile(r"```sh\n(.*?)```", re.S)
BLOCKS = re.compile(r"```[a-z]*\n.*?```", re.S)
INLINE = re.compile(r"`(loopmath flow [^`]+)`")
PLACEHOLDER = re.compile(r"[A-Z][A-Z_]*")
MERGE = "merge-extract-synthesize"


def run(*argv: str) -> tuple[int, str, str]:
    """`loopmath ARGV`: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, out.getvalue(), err.getvalue()


def emitted(*argv: str, code: int = 0) -> dict:
    got, out, err = run(*argv, "--json")
    assert got == code, (argv, err)
    return json.loads(out)


def examples() -> tuple[dict, dict]:
    """The topic's two JSON blocks: the answers, and the row."""
    answers, row = (json.loads(b) for b in re.findall(r"```json\n(.*?)```", TEXT, re.S))
    return answers, row


def names(after: str) -> list[str]:
    """The backticked names listed after `after` in the topic, outside parentheses, up to the end of the sentence
    or of the group `after` opens."""
    assert FLAT.count(after) == 1, after
    s, out, depth, i = FLAT[FLAT.index(after) + len(after):], [], 0, 0
    while i < len(s):
        c = s[i]
        if c == "`":
            j = s.index("`", i + 1)
            if depth == 0:
                out.append(s[i + 1:j])
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                break
        elif c == "." and depth == 0 and s[i + 1:i + 2] in ("", " "):
            break
        i += 1
    return out


def schema_list(out: dict) -> list[str]:
    """The members the topic lists for an output, after its `schema` and the schema name it gives."""
    return names(f"`schema` (`{out['schema']}`)")


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> dict:
    """Every command the topic teaches, run on the example: their JSON and their words."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
        mp.setattr("sys.stdin", io.StringIO(""))  # no terminal: `flow init` asks nothing
        return _journey(tmp_path_factory.mktemp("guide"))


def _journey(tmp: Path) -> dict:
    folder, home = tmp / "lit-brief", tmp / "home"
    at = ["--home", str(home)]
    answers, row = examples()
    answers_file = tmp / "answers.json"
    answers_file.write_text(json.dumps(answers), encoding="utf-8")
    out: dict = {"folder": folder, "said": []}

    def said(*argv: str) -> int:
        code, o, e = run(*argv)
        out["said"].append(o + e)
        return code

    out["init"] = emitted("flow", "init", str(folder), "--answers", str(answers_file), *at)
    out["inputs"] = json.loads((folder / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    out["show"] = emitted("flow", "show", str(folder))
    out["plan"] = emitted("flow", "plan", str(folder), *at)
    out["next"] = emitted("flow", "next", str(folder), "--n", "2", "--seed", "1", *at)

    rows = tmp / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out["record"] = emitted("flow", "record", str(folder), "--rows", str(rows), *at)
    out["score"] = emitted("flow", "score", str(folder), "--run", row["id"], "--judge", "person", "0.9", *at)
    out["status"] = emitted("flow", "status", str(folder), *at)
    out["check"] = emitted("flow", "check", str(folder), "--arm", MERGE, *at)
    out["error"] = emitted("flow", "check", str(folder), "--arm", "base", *at, code=1)
    out["plan_priced"] = emitted("flow", "plan", str(folder), *at)  # the recorded run has dollars; nothing written
    out["plan_refused"] = emitted("flow", "plan", str(folder), "--margin", "0.05", *at, code=1)
    broken = json.loads((folder / "flow.ocp.json").read_text(encoding="utf-8"))
    broken["flow"]["digest"] = "flw_000000000000"
    (tmp / "broken.ocp.json").write_text(json.dumps(broken), encoding="utf-8")
    out["show_refused"] = run("flow", "show", str(tmp / "broken.ocp.json"), "--json")

    # The refusals the topic quotes, each on its own input.
    bad = copy.deepcopy(answers)
    bad["steps"][2]["modle"] = "x"
    bad_file = tmp / "bad.json"
    bad_file.write_text(json.dumps(bad), encoding="utf-8")
    assert said("flow", "init", str(tmp / "bad"), "--answers", str(bad_file)) == 1
    tiny = copy.deepcopy(answers)
    tiny["objective"]["margin"] = 1e-6
    tiny_file = tmp / "tiny.json"
    tiny_file.write_text(json.dumps(tiny), encoding="utf-8")
    assert said("flow", "init", str(tmp / "tiny"), "--answers", str(tiny_file)) == 1

    def refused_row(name: str, change) -> None:
        r = copy.deepcopy(row)
        change(r)
        path = tmp / f"{name}.jsonl"
        path.write_text(json.dumps(r) + "\n", encoding="utf-8")
        assert said("flow", "record", str(folder), "--rows", str(path), *at) == 1, name

    refused_row("noend", lambda r: r.pop("ended_at"))
    refused_row("version", lambda r: r["flow"].update(version=2))
    refused_row("input", lambda r: r.update(task_id="q99"))
    refused_row("piece", lambda r: r["stages"][0].update(piece="nope"))
    refused_row("other", lambda r: r["stages"][0]["cost"].update(usd=0.99))
    out["refused"] = emitted("flow", "record", str(folder), "--rows", str(tmp / "noend.jsonl"), *at, code=1)
    doc = to_ocp(row, flow=load(folder))
    doc["run"]["flow"]["digest"] = "flw_000000000000"
    doc_file = tmp / "digest.ocp.json"
    doc_file.write_text(json.dumps(doc), encoding="utf-8")
    assert said("flow", "record", str(folder), "--ocp", str(doc_file), *at) == 1
    assert said("flow", "record", str(folder), "--rows", str(rows), *at) == 0
    assert said("flow", "score", str(folder), "--run", row["id"], "--judge", "person", "0.8", *at) == 1

    # Prefill: two logged runs of the example's steps, as rows and as run documents that declare the configuration.
    def put(name: str, value) -> str:
        (tmp / name).write_text(json.dumps(value), encoding="utf-8")
        return str(tmp / name)

    logged = tmp / "logged.jsonl"
    logged.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    plain = [put(f"plain-{i}.ocp.json", _plain(r)) for i, r in enumerate(ROWS)]
    stamped = [put(f"stamped-{i}.ocp.json", _stamped(_plain(r))) for i, r in enumerate(ROWS)]
    out["prefilled"] = {how: run("flow", "init", str(tmp / f"pf-{how}"), *given, "--json", *at)
                        for how, given in (("inferred", ["--from-rows", str(logged)]),
                                           ("declared", ["--from-ocp", *stamped]))}
    code, o, e = run("flow", "init", str(tmp / "pf"), "--from-rows", str(logged),
                     "--answers", put("inferred.json", INFERRED_ANSWERS), "--json", *at)
    assert (code, e) == (0, "")
    out["prefill_init"] = json.loads(o)
    out["prefill_ext"] = json.loads((tmp / "pf" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["ext"][
        EXT_PREFILL]
    for given in (["--from-rows", str(logged), "--answers", str(answers_file)],  # inferred: the steps in full
                  ["--from-ocp", *stamped, "--answers", str(answers_file)],  # declared: steps and checks given
                  ["--from-ocp", stamped[0], plain[1]]):  # a mix
        assert said("flow", "init", str(tmp / "pf-refused"), *given, *at) == 1
    return out


class _Built(Exception):
    pass


def _grab(self, *a, **k):
    raise _Built(self)


@pytest.fixture(scope="module")
def top() -> argparse.ArgumentParser:
    """The real top-level parser, as `cli.main` builds it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", _grab)
        with pytest.raises(_Built) as built:
            cli.main(["--help"])
    return built.value.args[0]


def _verbs(p: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(dict(a.choices) for a in p._actions if isinstance(a, argparse._SubParsersAction))


def _shown() -> list[str]:
    """Each `loopmath flow ...` command of the topic: sh block lines, then inline spans (a wrapped span joined)."""
    found = [line.strip() for block in SH.findall(TEXT) for line in block.splitlines()
             if line.strip().startswith("loopmath flow ")]
    return found + [" ".join(m.split()) for m in INLINE.findall(SH.sub("", TEXT))]


def test_every_flow_command_the_topic_shows_parses_with_the_real_parser(top):
    shown = _shown()
    assert len(shown) >= 20
    flow = _verbs(top)["flow"]
    for command in shown:
        words = shlex.split(command)[1:]
        flags = {s: a for a in _verbs(flow)[words[1]]._actions for s in a.option_strings}
        given = ["1" if PLACEHOLDER.fullmatch(w) and words[i - 1] in flags and flags[words[i - 1]].type in (int, float)
                 else w for i, w in enumerate(words)]  # a typed value's placeholder takes a number
        with redirect_stderr(io.StringIO()):
            try:
                args = top.parse_args(given)
            except SystemExit:
                pytest.fail(f"{command} does not parse")
        assert args.flow_command == words[1], command
    assert {shlex.split(c)[2] for c in shown} == set(FLOW_VERBS)


def test_the_last_section_names_every_flag_of_every_flow_command(top):
    section = TEXT[TEXT.index("## The flow commands"):]
    assert TEXT.rstrip().endswith(section.rstrip())  # the last section
    for verb, p in _verbs(_verbs(top)["flow"]).items():
        flags = {s for a in p._actions for s in a.option_strings if s.startswith("--") and s != "--help"}
        missing = {f for f in flags if f not in section}
        assert not missing, (verb, missing)


def test_the_example_answers_are_the_walks_and_each_answer_record_lists_its_members():
    answers, _ = examples()
    assert answers == walk.EXAMPLE
    assert re.findall(r"^- `([a-z_]+)`: ", TEXT[:TEXT.index("## 2.")], re.M) == list(walk.KEYS)
    for key, record in (("purpose", "purpose"), ("steps", "step"), ("checks", "checks"), ("judging", "judging"),
                        ("objective", "objective"), ("hypothesis", "hypothesis"), ("inputs", "inputs")):
        required, optional = walk._RECORDS[record]
        assert names(f"- `{key}`: ") == [*required, *optional], key
    for group, record in (("`gates` (each ", "gate"), ("`edits` (each ", "edit")):
        required, optional = walk._RECORDS[record]
        assert names(group) == [*required, *optional], group
    for member, default in walk.OBJECTIVE_DEFAULTS.items():
        assert f"`{member}` (default {f'`{default}`' if isinstance(default, str) else f'{default:.2f}'}" in FLAT


def test_each_member_list_is_what_the_command_emits(outputs):
    o = outputs
    for name in ("init", "show", "plan", "next", "record", "score", "status", "check", "error"):
        assert schema_list(o[name]) == list(o[name])[1:], name
    assert list(o["show"]) == list(SHOW_KEYS) and list(o["status"]) == list(STATUS_KEYS)
    assert schema_list(o["plan"]["plan"]) == list(o["plan"]["plan"])[1:]
    assert names("one line per id: ") == list(o["inputs"])
    spec = o["next"]["runs"][0]
    assert names("Each item of `runs` holds ") == list(spec)
    assert names("`person`, `flow` (") == list(spec["flow"])
    assert set(names("`settings` (per piece: ")) == {k for s in spec["settings"].values() for k in s}
    price = o["plan"]["plan"]["experiments"][0]["design"]["price"]
    assert names("`plan.experiments[].design.price` (") == list(price)
    assert names("`recorded` (each ") == list(o["record"]["recorded"][0])
    assert names("`fit` and `plan` (") == list(o["record"]["plan"]) == list(WRITER_PLAN_KEYS)
    assert list(o["refused"]) == ["schema", "flow", "refused"]
    assert names("`refused` (each ") == list(o["refused"]["refused"][0])
    s = o["status"]
    for after, row in (("`cost` (per step and model: ", s["cost"][0]),
                       ("`checks` (per check and arm: ", s["checks"][0]),
                       ("`quality` (per arm: ", s["quality"][0]), ("), `judges` (", s["judges"]),
                       ("`experiments` (each ", s["experiments"][0]), ("`dollars` (", s["dollars"])):
        assert names(after) == list(row), after
    assert names("`experiments` (each ") == list(STATE_KEYS)


def test_the_example_row_is_a_run_the_plan_asked_for_and_records_with_no_finding(outputs):
    _, row = examples()
    [spec] = [s for s in outputs["next"]["runs"] if s["run_id"] == row["id"]]
    assert (row["task_id"], row["arm"], row["flow"]) == (spec["input"], spec["arm"], spec["flow"])
    assert row["slate"] == {"id": spec["slate"], "members": spec["slate_members"]}
    plan = json.loads((outputs["folder"] / "plan.json").read_text(encoding="utf-8"))
    [arm] = [a for a in plan["arms"] if a["id"] == row["arm"]]
    pieces = arm["configuration"]["workflow"]["pieces"]
    assert [(p["role"], p["id"]) for p in pieces] == [(s["role"], s["piece"]) for s in row["stages"]]
    assert all(p.get("width", 1) == 1 for p in pieces)
    record = outputs["record"]
    assert [(r["run"], r["import"]) for r in record["recorded"]] == [(row["id"], "new")]
    assert record["findings"] == [] and record["plan"]["problem"] is None
    assert (outputs["score"]["state"], outputs["status"]["runs"]) == ("late", 1)
    assert outputs["check"]["verdict"] == "more" and outputs["error"]["exit"] == 1


def test_every_refusal_the_topic_quotes_is_the_commands_own_words(outputs):
    prose = " ".join(re.sub(r"`[^`]*`", "", BLOCKS.sub("", TEXT)).split())
    quoted = [q for q in re.findall(r'"([^"]+)"', prose) if q != "The flow commands"]
    assert len(quoted) >= 10
    said = " ".join(" ".join(outputs["said"]).split())
    template = "margin M: needs N check pairs at sd_pair S, above the limit of 1000000"
    assert template in quoted  # the planner's words, with its numbers as placeholders
    assert re.search(r"margin \S+: needs \d+ check pairs at sd_pair \S+, above the limit of 1000000", said)
    for q in quoted:
        assert q == template or q in said, q
    assert "## The flow commands" in TEXT


def test_the_prefill_summary_and_member_are_what_init_prints_and_writes(outputs):
    (code, out, err), (dcode, dout, derr) = (outputs["prefilled"][how] for how in ("inferred", "declared"))
    assert code == dcode == 2 and json.loads(out)["exit"] == json.loads(dout)["exit"] == 2
    assert json.loads(out)["error"] == err.splitlines()[-1][len("error: "):]
    first, *rest = err.splitlines()[:-1]
    for run_id in outputs["prefill_ext"]["from"]:
        first = first.replace(run_id, "RUN")
    assert re.search(r"```text\n(.*?)```", TEXT, re.S).group(1).splitlines() == [first, *rest]
    last = derr.splitlines()[-2]
    assert re.fullmatch(r"  configuration cfg_[0-9a-f]{12} as declared, with its checks and budget", last), last
    assert "`configuration ID as declared, with its checks and budget`" in FLAT
    assert list(outputs["prefill_init"]) == list(outputs["init"])  # `--json` names no prefill
    assert names('`flow.ext["dev.loopmath.prefill"]`: ') == list(outputs["prefill_ext"])
    assert outputs["prefill_ext"]["how"] == "inferred" and outputs["prefill_init"]["digest"] == outputs["init"][
        "digest"]


def test_the_plan_report_has_price_unknown_only_while_a_price_is_unknown(outputs):
    before, after = outputs["plan"], outputs["plan_priced"]  # no run recorded yet; then one with dollars
    assert set(before["price_unknown"]) == set(before["price_usd"]) and not any(before["price_usd"].values())
    assert "price_unknown" not in after and all(isinstance(v, float) for v in after["price_usd"].values())
    assert list(after) == [k for k in before if k != "price_unknown"]
    assert "`price_unknown` (only while a price is unknown)" in FLAT
    assert "once every price is known, the report has no `price_unknown`" in FLAT


def test_what_a_command_prints_under_json_when_it_exits_1_is_what_the_last_section_says(outputs):
    """Refused before its report, a command prints the error object; a refused new plan, a refused record and a
    document with an error finding print their own report in its place."""
    commands = FLAT[FLAT.index("## The flow commands"):]
    assert outputs["error"]["schema"] == "loopmath.error/1" and "`schema` (`loopmath.error/1`)" in commands
    refused = outputs["plan_refused"]
    assert list(refused) == list(outputs["plan_priced"]) and (refused["written"], refused["replaced"]) == (False, None)
    assert ("`flow plan`, refusing a new plan over recorded runs: the plan report, `written` false and `replaced` "
            "null;") in commands
    assert list(outputs["refused"]) == ["schema", "flow", "refused"]
    assert "`flow record`, refusing what it was given: `flow` and `refused`" in commands
    code, out, err = outputs["show_refused"]
    shown = json.loads(out)
    assert (code, err, list(shown)) == (1, "", ["schema", "path", "findings"])
    assert any(f["level"] == "error" for f in shown["findings"])
    assert "`flow show`, for a document with an error finding (nothing on stderr)" in commands
    assert "it cannot find: `path` and `findings`." in commands


def test_the_export_sentence_is_the_commands_own():
    assert FLOW_SAVING in FLAT


def test_the_notes_the_topic_names_are_the_reasons_flow_status_prints():
    """Each reason `flow status` ends a note with (`status.TAILS`, the scores naming no judge, the cost notes)."""
    start = TEXT.index("- `notes` are lines")
    bullet = TEXT[start:TEXT.index("\n- ", start)]
    named = re.findall(r"`([a-z-]+):(?:<[a-z ]+>|signals)`", bullet)
    assert sorted(named) == sorted({*TAILS, "unattributed", "cost"})


PINNED = ["calibrated", "fitted", "judges", "runs-n", "reference-n", "reference-seed"]


@pytest.mark.parametrize("name", PINNED)
def test_each_number_the_topic_states_is_the_codes_own(name, top):
    """The person scores before the judges are calibrated, the scored pairs before the spreads are fitted, the most
    judges a fit takes, and `flow next`'s defaults where the topic gives them, from the code and the real parser."""
    nxt = _verbs(_verbs(top)["flow"])["next"]
    words = {"calibrated": f"Until {CALIBRATED_AT} runs carry a person's score",
             "fitted": f"at assumed spreads until {FITTED_PAIRS} scored pairs exist",
             "judges": f"at most {MAX_SCORED_JUDGES} judges with readings per score",
             "runs-n": f"`--n N` is how many runs (default {nxt.get_default('n')}, one pair)",
             "reference-n": f"`--n` defaults to {nxt.get_default('n')},",
             "reference-seed": f"`--seed` to {nxt.get_default('seed')}."}[name]
    assert words in FLAT, words
