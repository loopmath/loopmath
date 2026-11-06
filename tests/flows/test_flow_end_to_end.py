"""The generic research flow end to end through the commands: `flow init` from the shipped answers, `flow plan`,
then two rounds of `flow next`, the simulator's own command on the specs it printed and `flow record`, a person's
score corrected with `flow score --replace`, then `flow status` and `flow check`. Each step is a command line read
for its exit code and its JSON; the only stand-in is the background refit, which is not started."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import simulate
from loopmath.flows.commands import SHOW_KEYS, WRITER_PLAN_KEYS
from loopmath.flows.status import STATUS_KEYS
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
ARM = "merge-extract-synthesize"  # the merge check: its pairs come after the screen's 32 runs
ROUNDS = ((1, 6), (2, 30))  # (seed, runs): the first round is screen runs only, the second reaches the merge check
INIT_KEYS = ["schema", "dir", "written", "kept", "flow", "digest", "plan"]
RECORD_KEYS = ["schema", "flow", "digest", "recorded", "findings", "advisories", "fit", "plan"]
SCORE_KEYS = ["schema", "run", "judge", "signal", "value", "replaced", "state", "fit", "plan"]
CHECK_KEYS = ["schema", "arm", "reference", "pairs", "p_quality", "p_cost", "threshold", "margin", "pair_cap",
              "verdict", "more_pairs", "extra"]


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


def flow(capsys, *argv: str) -> tuple[int, dict, str]:
    """`loopmath flow ARGV --json`: the exit code, the JSON on stdout and stderr."""
    code = cli.main(["flow", *argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def journey(tmp_path: Path, capsys) -> dict:
    """Every step's output, by name; the asserts on each step's own exit and members are made on the way."""
    folder, home = tmp_path / "lit-brief", tmp_path / "home"
    at = ["--home", str(home)]
    out: dict = {"folder": folder, "home": home, "rounds": []}

    code, out["init"], err = flow(capsys, "init", str(folder), "--answers", str(EXAMPLE / "answers.json"), *at)
    assert (code, err) == (0, "") and list(out["init"]) == INIT_KEYS
    assert (out["init"]["schema"], out["init"]["flow"]) == ("loopmath.flow-init/1", "lit-brief")
    assert sorted(Path(p).name for p in out["init"]["written"]) == ["flow.ocp.json", "plan.json", "questions.jsonl"]
    code, out["show"], err = flow(capsys, "show", str(folder))
    assert (code, err) == (0, "") and list(out["show"]) == list(SHOW_KEYS) and out["show"]["findings"] == []
    code, out["plan"], err = flow(capsys, "plan", str(folder), *at)
    assert (code, err) == (0, "") and out["plan"]["schema"] == "loopmath.flow-plan-report/1"
    plan_id = out["plan"]["plan"]["id"]
    assert plan_id == out["init"]["plan"]

    recorded: list[str] = []
    for seed, n in ROUNDS:
        code, specs, err = flow(capsys, "next", str(folder), "--seed", str(seed), "--n", str(n), *at)
        assert (code, err) == (0, "") and specs["schema"] == "loopmath.flow-next/1" and specs["plan"] == plan_id
        ids = [s["run_id"] for s in specs["runs"]]
        assert len(ids) == n and not set(ids) & set(recorded)
        spec_file = tmp_path / f"specs-{seed}.json"
        spec_file.write_text(json.dumps(specs), encoding="utf-8")
        runs_dir = tmp_path / f"runs-{seed}"
        assert simulate.main([str(folder), "--specs", str(spec_file), "--seed", str(seed),
                              "--truth", str(EXAMPLE / "truth.json"), "--out", str(runs_dir)]) == 0
        capsys.readouterr()
        files = sorted(runs_dir.glob("*.ocp.json"))
        docs = {json.loads(p.read_text(encoding="utf-8"))["run"]["id"]: json.loads(p.read_text(encoding="utf-8"))
                for p in files}
        assert sorted(docs) == sorted(ids)
        for spec in specs["runs"]:  # the run the spec asked for: its input, slate, arm and judges
            run = docs[spec["run_id"]]["run"]
            assert run["flow"] == spec["flow"] and run["slate"]["id"] == spec["slate"]
            scored = {s["source"]["ref"] for s in run["signals"] if s.get("kind") == "score"}
            assert scored == set(spec["judges"]) | ({"person"} if spec["person"] else set())

        code, record, err = flow(capsys, "record", str(folder), "--ocp", *map(str, files), "--allow-simulated", *at)
        assert code == 0 and list(record) == RECORD_KEYS and record["schema"] == "loopmath.flow-record/1"
        assert [r["run"] for r in record["recorded"]] == sorted(ids) and {r["import"] for r in record["recorded"]} == {
            "new"}
        assert list(record["plan"]) == list(WRITER_PLAN_KEYS)
        assert (record["plan"]["id"], record["plan"]["advanced"], record["plan"]["problem"]) == (plan_id, True, None)
        recorded += ids
        out["rounds"].append({"specs": specs, "files": files, "record": record})
    out["recorded"] = recorded

    marked = [s["run_id"] for s in out["rounds"][0]["specs"]["runs"] if s["person"]]
    assert marked, "the first round asks for no person score"
    out["marked"] = marked[0]
    return out


def _later(stamp: str) -> str:
    when = datetime.fromisoformat(stamp.replace("Z", "+00:00")) + timedelta(hours=1)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_a_flow_from_init_to_check_through_the_commands(tmp_path, capsys):
    out = journey(tmp_path, capsys)
    folder, home = out["folder"], out["home"]
    at = ["--home", str(home)]

    # A simulated run already stored is still refused without the flag, by name.
    files = out["rounds"][0]["files"]
    code, refused, err = flow(capsys, "record", str(folder), "--ocp", *map(str, files), *at)
    assert code == 1 and refused["schema"] == "loopmath.flow-record/1" and not refused.get("recorded")
    assert sorted(r["run"] for r in refused["refused"]) == sorted(out["rounds"][0]["specs"]["runs"][i]["run_id"]
                                                                  for i in range(len(files)))
    assert all(r["reason"].endswith("--allow-simulated records simulated runs") for r in refused["refused"])

    # A person's score of a marked run: the simulator wrote one, so a second is refused; --replace corrects it.
    run = out["marked"]
    [old] = [s for s in Store(home).run_doc(run)["run"]["signals"]
             if s.get("kind") == "score" and s["source"].get("ref") == "person"]
    code, before, err = flow(capsys, "status", str(folder), *at)
    code, refusal, err = flow(capsys, "score", str(folder), "--run", run, "--judge", "person", "0.95", *at)
    assert code == 1 and refusal["schema"] == "loopmath.error/1"
    assert refusal["error"].startswith(f"run {run} already has a score of judge 'person' under 'quality' (")
    assert refusal["error"].endswith("); --replace adds this one as the newest reading")
    value = "0.05" if old["value"] > 0.5 else "0.95"
    code, score, err = flow(capsys, "score", str(folder), "--run", run, "--judge", "person", value, "--replace",
                            "--observed-at", _later(old["observed_at"]), *at)
    assert code == 0 and list(score) == SCORE_KEYS and score["schema"] == "loopmath.flow-score/1"
    assert (score["run"], score["judge"], score["value"], score["state"]) == (run, "person", float(value), "late")
    assert score["replaced"] == [old["id"]] and score["plan"]["problem"] is None

    # Status: the recorded runs, their dollars, and the corrected reading in the quality of the run's arm.
    code, status, err = flow(capsys, "status", str(folder), *at)
    assert code == 0 and list(status) == list(STATUS_KEYS) and status["schema"] == "loopmath.flow-status/1"
    assert (status["runs"], status["other_revisions"], status["plan_problem"]) == (len(out["recorded"]), 0, None)
    assert status["dollars"]["runs"] == len(out["recorded"]) and status["dollars"]["usd"] > 0
    arm = next(r["arm"] for r in out["rounds"][0]["record"]["recorded"] if r["run"] == run)
    pooled = {s["arm"]: s["pooled_mean"] for s in status["quality"]}
    assert pooled[arm] != {s["arm"]: s["pooled_mean"] for s in before["quality"]}[arm]

    # Check: the merge arm, its pairs counted from the recorded runs.
    specs = [s for r in out["rounds"] for s in r["specs"]["runs"] if s["experiment"] == ARM]
    slates = [s["slate"] for s in specs]
    pairs = sum(1 for slate in set(slates) if slates.count(slate) == 2)
    assert pairs >= 1
    code, check, err = flow(capsys, "check", str(folder), "--arm", ARM, *at)
    assert code == 0 and list(check) == CHECK_KEYS and check["schema"] == "loopmath.flow-check/1"
    assert (check["arm"], check["reference"], check["pairs"]) == (ARM, "base", pairs)
    assert check["verdict"] in ("adopt", "reject", "more") and check["pair_cap"] == 24
