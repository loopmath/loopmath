"""`loopmath runs` polish: `runs RUN`, the configuration in the table, `--wide`, plurals and units."""

from __future__ import annotations

import pytest
from store_helpers import cli

from loopmath.store import Store

CFG = "cfg_b214f518e85f"


def _doc(run: str, *, model: str, effort: str, perf: float, started: str) -> dict:
    """A designed solo run with a heldout_perf score, shaped like a converted benchmark run (synthetic values)."""
    return {
        "ocp": "0.3", "producer": {"name": "synthetic", "version": "1"}, "privacy": {"profile": "metadata_only"},
        "run": {
            "id": run, "started_at": started, "ended_at": started,
            "task": {"id": f"tsk_{run}", "type": "feature", "repo": "bench", "subtype": "p1"},
            "configuration": {
                "id": CFG if model == "sol" else "cfg_000000000001", "source": "designed",
                "workflow": {"id": "solo", "version": 1, "pieces": [{"id": "implement", "role": "implementer", "width": 1}],
                             "artifacts": [{"id": "diff", "kind": "diff"}], "edges": [["implement", "diff"]]},
                "settings": {"implement": {"harness": "codex", "model": {"raw": f"gpt-5.6-{model}", "id": f"gpt-5.6-{model}"},
                                           "effort": effort}},
            },
            "acceptance_rule": {"name": "perf", "definition": "heldout_perf>=2400", "requires": [],
                                "score": {"name": "heldout_perf", "target": 2400, "better": "higher"}},
            "signals": [{"id": f"sig_{run}", "run": run, "kind": "score", "name": "heldout_perf", "value": perf,
                         "unit": "perf", "better": "higher", "observed_at": started, "tier": "verified",
                         "source": "judge"}],
        },
        "nodes": [{"id": "implement", "kind": "impl"}],
        "attempts": [{"id": f"att_{run}", "node": "implement", "n": 1, "status": "done", "started_at": started,
                      "ended_at": started, "cost": {"usd": 1.25, "input_tokens": 100, "output_tokens": 10, "basis": "measured"}}],
    }


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "lm"
    store = Store(h)
    store.import_run(_doc("rq-a02", model="sol", effort="xhigh", perf=3140, started="2026-09-22T18:16:00-07:00"),
                     finished=True, check=False)
    store.import_run(_doc("rq-a06", model="luna", effort="max", perf=1875, started="2026-09-23T02:45:00-07:00"),
                     finished=True, check=False)
    return h


def test_runs_positional_id_is_runs_dash_dash_run(capsys, home):
    code, by_flag, err = cli(capsys, home, "runs", "--run", "rq-a02")
    assert code == 0, err
    code, by_position, err = cli(capsys, home, "runs", "rq-a02")
    assert code == 0, err
    assert by_position == by_flag and by_position.startswith("run rq-a02 ")
    code, _, err = cli(capsys, home, "runs", "rq-a02", "--run", "rq-a06")
    assert code == 1 and "give one" in err
    code, _, _ = cli(capsys, home, "runs", "rq-a02", "--run", "rq-a02")
    assert code == 0
    code, _, err = cli(capsys, home, "runs", "nope")
    assert code == 2 and "no run nope" in err


def test_the_table_names_model_and_effort_per_arm(capsys, home):
    code, out, err = cli(capsys, home, "runs")
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0].startswith("2 runs, ")
    rows = {line.split()[0]: line for line in lines[2:]}
    assert "solo: gpt-5.6-sol/xhigh" in rows["rq-a02"] and "solo: gpt-5.6-luna/max" in rows["rq-a06"]
    assert CFG not in out  # the cfg_ id only with --wide (and in --json)
    code, wide, err = cli(capsys, home, "runs", "--wide")
    assert code == 0, err
    assert "config" in wide.splitlines()[1] and CFG in wide
    code, obj, err = cli(capsys, home, "runs", "--json")
    assert {r["config"]["id"] for r in obj["runs"]} == {CFG, "cfg_000000000001"}


def test_plurals_and_no_repeated_units(capsys, home):
    code, out, err = cli(capsys, home, "runs", "--since", "2026-09-23")
    assert code == 0, err
    assert out.splitlines()[0].startswith("1 run, ")
    assert "heldout_perf 1875" in out and "1875 perf" not in out
    code, out, err = cli(capsys, home, "runs", "rq-a02")
    assert code == 0, err
    assert "over 1 round" in out and "round(s)" not in out
    assert "heldout_perf 3140" in out and "3140 perf" not in out
    assert "score heldout_perf = 3140 (verified)" in out


def test_a_unit_the_name_does_not_say_is_kept():
    from loopmath.views.runs import _unit_suffix

    assert _unit_suffix("heldout_score", "points") == " points"
    assert _unit_suffix("heldout_perf", "perf") == ""
    assert _unit_suffix("runtime_s", "s") == "" and _unit_suffix("runtime", "s") == " s"
    assert _unit_suffix("quality", None) == ""
