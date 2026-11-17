"""`loopmath guide eval`, the eval topic (0.4.5), against the commands it teaches.

Every `loopmath eval ...` line in its sh blocks parses with the real parser, and the blocks walk every stage. Its
last section, "The eval commands", names every eval command and each flag the parser gives that command. The
two refusal forms and the exit codes it states are what a refused import, a failed study check, a missing fit and a
mistyped command give. The anchor states it gives, and the counts where they change, are what `eval scores check` and
`eval fit --check-only` report on a study with a declared person who scored 5, 6, 29 or 30 outputs (and 9 or 10 under
`min_anchor_targets` 10), and `eval judges` measures a bias against the person on exactly the dimensions it says.
On one fit that did not converge, a pilot plan is drawn and the commands it says refuse do. The catalog it describes
is the one `eval simulate --list` prints, and the specs it says write a value file are the ones that do. Its practice
block runs as written up to the sampler, and the fit options its limits call not built are refused. The skill tests
(`tests/skill/test_skill_guide.py`, `test_skill_lint.py`) check the topic list, the cross references and that every
flag a guide shows is in its reference. Commands run in this process on synthetic studies under tmp_path; nothing
samples.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.eval import commands, judges, specs, testing
from loopmath.eval import contracts as C
from loopmath.skill import guide

TEXT = guide.text("eval")
FLAT = " ".join(TEXT.split())  # a wrapped line reads as one
HEADING = "## The eval commands"
SECTION = TEXT.split(f"\n{HEADING}\n", 1)[1]
SH = re.compile(r"```sh\n(.*?)```", re.S)
# The stages the walk goes through; `scores add` and `scores export` are in the last section only.
WALK = ("rubric check", "scores import", "scores check", "fit", "judges", "value", "steps", "labels next", "simulate")
# Flags every command that reads a fit file takes; the last section names them once, after the list.
FIT_READER = {"--study", "--strict", "--allow-unconverged"}


def lm(*argv: object) -> tuple[int, str, str]:
    """`loopmath ARGV` in this process: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([str(word) for word in argv])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def parser() -> argparse.ArgumentParser:
    """The real top-level parser, as `cli.main` builds it."""
    class Built(Exception):
        pass

    def grab(self, *args, **kwargs):
        raise Built(self)

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def sub(p: argparse.ArgumentParser, word: str) -> argparse.ArgumentParser:
    [action] = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
    return action.choices[word]


def shown() -> list[str]:
    """Every `loopmath eval ...` line of the topic's sh blocks."""
    return [line.strip() for block in SH.findall(TEXT) for line in block.splitlines()
            if line.strip().startswith("loopmath eval ")]


def has_flag(text: str, flag: str) -> bool:
    """`flag` as a whole word in `text`: `--run` is not found in `--runs`."""
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None


OBSERVED = "2026-11-16T21:00:00-08:00"


def study(root: Path, human: int = 0, min_anchor: int | None = None) -> dict:
    """A synthetic study: one linear component, 40 runs with neutral ids (20 inputs, 2 arms), two model judges
    with two calls on every run, and a person, declared as the anchor, who scored the first `human` runs once."""
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "guide", "version": "1", "components": [
        {"id": "quality", "title": "Quality", "scale": {"kind": "linear", "range": [0, 10]}}]})
    declared = {"schema": C.SCHEMA_JUDGES, "anchor": ["reader"], "judges": [
        {"id": "reader", "kind": "person"},
        {"id": "j_one", "kind": "model", "model": {"id": "model_one", "family": "fam_one"}},
        {"id": "j_two", "kind": "model", "model": {"id": "model_two", "family": "fam_two"}}]}
    if min_anchor is not None:
        declared["min_anchor_targets"] = min_anchor
    people = C.validate_judges(declared)
    runs = [C.validate_run({"schema": C.SCHEMA_RUNS, "run": f"r{2 * i + a + 1:03d}", "input": f"i{i:03d}", "arm": arm,
                            "steps": [{"step": "final", "model": {"id": f"writer_{a}", "family": "writers"}}]})
            for i in range(20) for a, arm in enumerate(("control", "candidate"))]
    testing.write_study_files(root, rubric, people, runs)
    reading = {"schema": C.SCHEMA_SCORES, "step": "final", "component": "quality", "rubric": "guide@1",
               "observed_at": OBSERVED}
    rows = [{**reading, "id": f"s_{r['run']}_{j}_{k}", "run": r["run"], "judge": j, "call": f"c_{r['run']}_{j}_{k}",
             "value": value} for r in runs for j, value in (("j_one", 6), ("j_two", 8)) for k in (1, 2)]
    rows += [{**reading, "id": f"h_{r['run']}", "run": r["run"], "judge": "reader", "call": f"h_{r['run']}", "value": 5}
             for r in runs[:human]]
    C.write_lines(root / "scores.jsonl", C.SCHEMA_SCORES, rows)
    return {"root": root, "rubric": rubric, "judges": people, "runs": runs}


def test_every_eval_command_in_the_blocks_parses_and_the_walk_covers_each_stage(capsys):
    top, lines = parser(), shown()
    assert len(lines) >= len(WALK), lines
    for line in lines:
        try:
            top.parse_args(shlex.split(line)[1:])
        except SystemExit:
            raise AssertionError(f"does not parse: {line}: {capsys.readouterr().err.strip()[-300:]}")
    for stage in WALK:
        assert any(line.startswith(f"loopmath eval {stage} ") for line in lines), stage


def test_the_last_section_names_every_eval_command_and_each_of_its_flags():
    assert re.findall(r"^## .+$", TEXT, re.M)[-1] == HEADING
    bullets = [b.split("\n\n", 1)[0] for b in SECTION.split("\n- ")[1:]]
    after = SECTION.rstrip("\n").split("\n\n")[-1]  # the paragraph after the list
    eval_parser = sub(parser(), "eval")
    problems = []
    for words in commands.COMMANDS:
        p = eval_parser
        for word in words.split():
            p = sub(p, word)
        mine = [b for b in bullets if b.startswith(f"`loopmath eval {words} ")]
        if len(mine) != 1:
            problems.append(f"eval {words}: {len(mine)} bullets")
            continue
        flags = {s for a in p._actions for s in a.option_strings if s.startswith("--") and s != "--help"}
        for flag in sorted(flags):
            shared = flag in FIT_READER and f"`eval {words}`" in after and has_flag(after, flag)
            if not (has_flag(mine[0], flag) or shared):
                problems.append(f"eval {words}: {flag} is not named")
    assert not problems, "\n".join(problems)
    assert len(bullets) == len(commands.COMMANDS)  # one bullet per command, and no other


def test_both_refusal_forms_and_a_usage_error_exit_as_the_last_section_says(tmp_path):
    flat = " ".join(SECTION.split())
    assert "1 a refusal" in FLAT and "2 usage" in FLAT

    # The fallback object: a fit file that is missing.
    code, out, err = lm("eval", "judges", tmp_path / "missing" / "fit.json", "--json")
    assert code == 1 and err.startswith("error: ")
    got = json.loads(out)
    assert list(got) == ["schema", "ok", "command", "error", "exit"]
    assert got["schema"] == "loopmath.error/1" and got["ok"] is False and got["exit"] == 1
    assert "`schema` (`loopmath.error/1`), `ok` (false), `command`, `error` and `exit`" in flat
    assert f"`{judges.SCHEMA}`" in SECTION  # the schema it gives as an example is the judges report's

    # The import's own report: one CSV reading with no judge refuses the whole import and writes nothing.
    s = study(tmp_path / "s")
    out_dir = tmp_path / "imported"
    (tmp_path / "bad.csv").write_text("run,step,component,judge,call,value,observed_at\n"
                                      f"r001,final,quality,,c1,7,{OBSERVED}\n"
                                      f"r002,final,quality,j_one,c2,7,{OBSERVED}\n", encoding="utf-8")
    argv = ("eval", "scores", "import", "--csv", tmp_path / "bad.csv", "--rubric", s["root"] / "rubric.json",
            "--out", out_dir)
    code, out, err = lm(*argv, "--json")
    got = json.loads(out)
    assert code == 1 and err == "" and got["schema"] == "loopmath.eval-scores-import/1" and got["ok"] is False
    assert len(got["refused"]) == 1 and list(got["refused"][0]) == ["table", "id", "source", "reason"]
    assert got["written"] == [] and not (out_dir / "scores.jsonl").exists()
    assert "`refused` rows (each with `table`, `id`, `source`, `reason`) and writes nothing" in flat
    code, out, err = lm(*argv)
    assert code == 1 and err.startswith("error: ") and "judge" in err

    # The check's own report: a value off the scale.
    rows = [json.loads(line) for line in (s["root"] / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["value"] = 11
    (s["root"] / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    argv = ("eval", "scores", "check", s["root"], "--rubric", s["root"] / "rubric.json", "--judges",
            s["root"] / "judges.json")
    code, out, err = lm(*argv, "--json")
    got = json.loads(out)
    assert code == 1 and err == "" and got["schema"] == "loopmath.eval-scores-check/1" and got["ok"] is False
    assert [p["code"] for p in got["problems"]] == ["bad_value"]
    assert list(got["problems"][0]) == ["file", "where", "problem", "fix", "code", "message"]
    assert "`problems` (each with `file`, `where`, `problem`, `fix`, `code`, `message`)" in flat
    code, out, err = lm(*argv)
    assert code == 1 and err.startswith("error: ") and "outside the scale" in err

    # A usage error.
    code, _, err = lm("eval", "fit", tmp_path)  # no --rubric, no --judges
    assert code == 2 and "--rubric" in err


# What step 3 says, rule by rule: (count below which, state) with the default `min_anchor_targets`.
ANCHOR_RULES = ("below `min_anchor_targets` (6 unless `judges.json` sets another number): `relative`",
                "from `min_anchor_targets` to 29: `weak`", "30 or more (and at least `min_anchor_targets`): `anchored`")


@pytest.mark.parametrize("human, min_anchor, state", [
    (0, None, "relative"), (5, None, "relative"), (6, None, "weak"), (29, None, "weak"), (30, None, "anchored"),
    (9, 10, "relative"), (10, 10, "weak")])
def test_a_dimension_is_anchored_on_the_person_only_from_the_counts_step_3_gives(tmp_path, human, min_anchor, state):
    """A person is declared in every case; what the dimension is anchored on depends on how many outputs they
    scored, as `eval scores check` and `eval fit --check-only` both report."""
    flat = " ".join(TEXT.split())
    assert all(rule in flat for rule in ANCHOR_RULES)
    s = study(tmp_path, human=human, min_anchor=min_anchor)
    root = s["root"]
    code, out, err = lm("eval", "fit", root, "--rubric", root / "rubric.json", "--judges", root / "judges.json",
                        "--check-only", "--json")
    assert code == 0, err
    anchor = json.loads(out)["anchor"]
    assert [(row["dimension"], row["state"], row["n_targets"]) for row in anchor["dimensions"]] == [
        ("final/quality", state, human)]
    assert anchor["kind"] == ("judges_mean" if state == "relative" else "person")  # "`anchor: person` only says..."
    code, out, err = lm("eval", "fit", root, "--rubric", root / "rubric.json", "--judges", root / "judges.json",
                        "--check-only")
    assert code == 0 and f"anchor: {anchor['kind']}" in out.splitlines()
    assert f"  final/quality: {state} on {human} anchored targets" in out.splitlines()
    code, out, err = lm("eval", "scores", "check", root, "--rubric", root / "rubric.json", "--judges",
                        root / "judges.json", "--json")
    assert code == 0, err
    rows = [row for row in json.loads(out)["design"] if row["term"] == "anchor"]
    assert [(row["dimension"], row["state"], row["count"]) for row in rows] == [("final/quality", state, human)]


def test_eval_judges_measures_a_bias_against_the_person_only_on_weak_and_anchored_dimensions(tmp_path):
    """One fit, three dimensions in three anchor states (the fit file's anchor block set by hand, as `eval fit`
    would write it): `relative_to` is `anchor` on the weak and the anchored one, `judges_mean` on the relative one."""
    testing.write_study_files(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json")
    fit = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    rows = fit["anchor"]["dimensions"]
    assert fit["anchor"]["kind"] == "person" and len(rows) >= 3
    states = {rows[0]["dimension"]: ("relative", 5), rows[1]["dimension"]: ("weak", 6)}
    for row in rows:
        row["state"], row["n_targets"] = states.get(row["dimension"], ("anchored", 30))
    (tmp_path / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    code, out, err = lm("eval", "judges", built.path, "--json")
    assert code == 0, err
    seen = {(d["dimension"], d["bias"]["relative_to"], d["bias"]["anchor_state"])
            for j in json.loads(out)["judges"] for d in j["dims"] if d["bias"]}
    want = {"relative": "judges_mean", "weak": "anchor", "anchored": "anchor"}
    assert seen and all(relative_to == want[state] for _, relative_to, state in seen)
    assert {state for _, _, state in seen} == {"relative", "weak", "anchored"}
    assert "`relative_to` (`anchor` or `judges_mean`) and `anchor_state`" in FLAT


# The commands step 4 says refuse a fit that did not converge, and the pilot plan that still takes it.
REFUSING = ("`eval judges`, `eval value`, `eval steps`, `eval rubric check --fit` and `eval labels next --mode active` "
            "refuse it")
PILOT = "`eval labels next --mode pilot` takes an unconverged fit without `--allow-unconverged`"


def test_a_pilot_plan_takes_an_unconverged_fit_and_the_numerical_reports_and_active_mode_refuse_it(tmp_path):
    assert REFUSING in FLAT and PILOT in FLAT
    s = study(tmp_path)
    root = s["root"]
    bad = testing.build_fit(root / "fit.json", rubric=s["rubric"], judges=s["judges"], runs=s["runs"],
                            converged=False)
    assert bad.fit["diagnostics"]["converged"] is False
    (root / "value.json").write_text(json.dumps({"schema": C.SCHEMA_VALUE, "weights": {"quality": 1}, "gates": [],
                                                 "objective": {"kind": "best_value", "reference": "control"}}),
                                     encoding="utf-8")
    code, out, err = lm("eval", "labels", "next", bad.path, "--mode", "pilot", "--n", 30, "--out",
                        root / "labels-001.json", "--form", root / "labels-001.csv", "--json")
    assert code == 0, err
    got = json.loads(out)
    rows = got["plan"]["assignments"]  # drawn with a probability each, so the count is near 30, not exactly 30
    assert "pilot" in {row["kind"] for row in rows} <= {"pilot", "repeat"} and (root / "labels-001.csv").exists()
    assert any("did not converge" in warning for warning in got["warnings"])
    for argv in (("eval", "judges", bad.path), ("eval", "value", bad.path, "--value", root / "value.json"),
                 ("eval", "steps", bad.path, "--value", root / "value.json"),
                 ("eval", "rubric", "check", root / "rubric.json", "--fit", bad.path),
                 ("eval", "labels", "next", bad.path, "--mode", "active", "--n", 30)):
        code, out, err = lm(*argv, "--json")
        got = json.loads(out)
        assert code == 1 and got["schema"] == "loopmath.error/1", (argv, code, err)
        assert "did not converge" in got["error"] and "--allow-unconverged" in got["error"], argv


def test_the_catalog_is_the_one_simulate_lists_and_the_named_specs_write_a_value_file(tmp_path):
    code, out, _ = lm("eval", "simulate", "--list")
    listed = [line.split()[0] for line in out.splitlines() if line.strip()]
    assert code == 0 and "linear" in listed
    assert f"The catalog has {len(listed)} specs" in FLAT
    with_value = set()
    for name in listed:  # each spec on 12 inputs: the files it writes do not depend on the size
        spec = tmp_path / f"{name}.spec.json"
        spec.write_text(json.dumps(specs.get(name, inputs=12)), encoding="utf-8")
        assert lm("eval", "simulate", spec, "--out", tmp_path / name, "--seed", 0)[0] == 0, name
        if (tmp_path / name / "value.json").exists():
            with_value.add(name)
    said = re.search(r"; (`[a-z-]+`(?:, `[a-z-]+`)* and `[a-z-]+`) also write a `value\.json`", FLAT)
    assert said and set(re.findall(r"`([a-z-]+)`", said.group(1))) == with_value


def test_the_practice_block_runs_as_written_up_to_the_sampler(tmp_path, monkeypatch):
    """The block after "Try it without data", with `--check-only` on its fit so nothing samples; without
    `--allow-simulated` the fit is refused, and the fit options the limits call not built are refused by name."""
    monkeypatch.chdir(tmp_path)
    block = SH.findall(TEXT.split("## Try it without data", 1)[1])[0]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert [line.split()[2] for line in lines] == ["simulate", "simulate", "fit"]
    for line in lines[:2]:
        assert lm(*shlex.split(line)[1:])[0] == 0, line
    fit = shlex.split(lines[2])[1:]
    before = sorted(p.name for p in Path("study").iterdir())
    code, out, err = lm(*fit, "--check-only")
    assert code == 0 and out.startswith("design: ") and sorted(p.name for p in Path("study").iterdir()) == before
    code, _, err = lm(*[w for w in fit if w != "--allow-simulated"], "--check-only")
    assert code == 1 and "--allow-simulated" in err
    for flag, value in (("--halo", "full"), ("--cuts", "free")):
        assert f"`eval fit {flag} {value}`" in FLAT or f"`{flag} {value}`" in FLAT
        code, _, err = lm(*fit, "--check-only", flag, value)
        assert code == 1 and "not built" in err, (flag, err)


def test_eval_record_sits_after_eval_and_each_sends_the_agent_to_the_other():
    topics = list(guide.TOPICS)
    assert topics[-3:] == ["eval", "eval-record", "reference"]
    assert "`loopmath guide eval-record`" in TEXT and "`loopmath guide eval`" in guide.text("eval-record")
    assert TEXT.splitlines()[0].startswith("# ") and TEXT.splitlines()[0].endswith("(experimental)")
    for topic in ("eval", "eval-record"):
        assert guide.TOPICS[topic].endswith("(experimental)")
        assert guide.text(topic).startswith("# ") and "experimental" in guide.text(topic).lower()
