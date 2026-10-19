"""One `loopmath.harness-runs/0.1` row becomes one strict-valid OCP v0.3 run (plan section 3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from loopmath import harness, harness_runs as HR
from loopmath.ocp.conformance import validate_doc
from loopmath.ocp.emit import validate_strict
from loopmath.price import load_prices, price_run
from loopmath.types import Setting
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import make_config

NS = "dev.loopmath.adapter.harness-runs"
WHEN = "2026-10-18T17:14:02-07:00"


def _errors(doc: dict) -> list:
    return [f for f in validate_doc(doc) if f.level == "error"]


def _strict_errors(doc: dict) -> list:   # the store's own check: OCP 0.3 declared
    return [f for f in validate_strict(doc) if f["level"] == "error"]


def _tree(root: Path, note: str) -> Path:
    (root / ".claude").mkdir(parents=True)
    (root / "CLAUDE.md").write_text(f"Be brief.\n{note}\n", encoding="utf-8")
    (root / ".claude" / "settings.json").write_text('{"enabledPlugins": {"lint@acme": true}}\n', encoding="utf-8")
    return root


def _row(**over) -> dict:
    row = {"schema": HR.SCHEMA, "task_id": "ledger-1", "arm": "base", "agent": "opencode",
           "model": "gpt-5.4", "effort": "high", "started_at": "2026-10-18T17:10:00-07:00", "ended_at": WHEN,
           "tool": {"name": "harness2", "version": "0.1.0", "run_id": "jb-oc-agg2/ledger-1/base1",
                    "task_id": "ledger-1", "stage": "base1"}}
    row.update(over)
    return row


def test_one_row_is_one_strict_valid_v03_run():
    doc = HR.to_ocp(_row(outcome={"score": 0.75, "judge": "judge-x", "scale": "fraction"},
                         cost={"usd": 0.42, "recovered": True, "usd_source": "opencode"}))
    assert doc["ocp"] == "0.3"
    assert _errors(doc) == []
    run = doc["run"]
    assert run["task"]["id"] == "ledger-1"
    assert run["provenance"]["kind"] == "logged"
    assert run["labels"]["arm"] == "base"
    assert doc["producer"]["name"].startswith("loopmath")
    assert len(doc["attempts"]) == 1


def test_the_configuration_is_the_solo_workflow_with_the_rows_setting():
    doc = HR.to_ocp(_row())
    solo = catalog()["solo"]
    want = make_config(solo, {solo.pieces[0].id: Setting(harness="opencode", model="gpt-5.4", effort="high")})
    assert doc["run"]["configuration"]["id"] == want.id
    other = HR.to_ocp(_row(model="gpt-5.4-mini"))
    assert other["run"]["configuration"]["id"] != want.id


def test_the_harness_record_sits_at_ext_key_from_a_tree_or_inline(tmp_path):
    tree = _tree(tmp_path / "rows" / "trees" / "a", "A")
    parent = _tree(tmp_path / "rows" / "trees" / "base", "base")
    by_path = HR.to_ocp(_row(harness="trees/a"), base_dir=tmp_path / "rows")
    rec = by_path["run"]["ext"][harness.EXT_KEY]
    assert harness.BARE_KEY not in by_path["run"]["ext"]
    assert rec["digest"] == harness.digest(tree)
    assert rec["source"] == "harness2" and rec["tool"]["stage"] == "base1"
    assert harness.factors(rec) == harness.factors(harness.record_from_tree(tree))
    inline = harness.record_from_tree(tree, source="harness2", parent=harness.digest(parent),
                                      tool={"name": "harness2", "stage": "inv1"})
    inline["future_field"] = {"kept": True}
    by_record = HR.to_ocp(_row(harness=copy.deepcopy(inline)))
    got = by_record["run"]["ext"][harness.EXT_KEY]
    assert got["digest"] == inline["digest"] and got["parent"] == harness.digest(parent)
    assert got["future_field"] == {"kept": True}
    assert harness.run_record(by_record)["digest"] == inline["digest"]
    assert harness.EXT_KEY not in HR.to_ocp(_row(harness=None))["run"]["ext"]


def test_a_tree_path_may_not_leave_the_rows_folder(tmp_path):
    (tmp_path / "rows").mkdir()
    _tree(tmp_path / "outside", "x")
    with pytest.raises(ValueError, match="harness"):
        HR.to_ocp(_row(harness="../outside"), base_dir=tmp_path / "rows")


def test_a_score_becomes_a_score_signal_with_its_judge():
    doc = HR.to_ocp(_row(outcome={"score": 0.75, "judge": "judge-x", "scale": "fraction",
                                  "n_passed": 3, "n_criteria": 4}))
    [sig] = doc["run"]["signals"]
    assert (sig["kind"], sig["name"], sig["value"]) == ("score", "judge", 0.75)
    assert sig["source"] == {"kind": "judge", "ref": "judge-x"}
    assert sig["scale"] == "fraction" and sig["better"] == "higher"
    assert sig["observed_at"] == WHEN
    assert sig["ext"][NS]["n_passed"] == 3


def test_a_null_score_is_declared_never_zero():
    doc = HR.to_ocp(_row(outcome={"score": None, "judge": "judge-x"}))
    [sig] = doc["run"]["signals"]
    assert sig["kind"] == "score" and sig["value"] is None
    assert _errors(doc) == []


def test_unscored_has_no_signal():
    for outcome in (None, "absent"):
        row = _row() if outcome == "absent" else _row(outcome=None)
        assert not HR.to_ocp(row)["run"].get("signals")


def test_a_verdict_becomes_the_accepted_signal_and_rule():
    doc = HR.to_ocp(_row(outcome={"accepted": False, "by": "referee"}))
    [sig] = doc["run"]["signals"]
    assert (sig["kind"], sig["name"], sig["value"]) == ("verdict", "accepted", "reject")
    assert doc["run"]["acceptance_rule"]["requires"] == ["accepted"]


def test_accept_at_writes_a_score_rule():
    doc = HR.to_ocp(_row(outcome={"score": 1.0, "judge": "j", "scale": "fraction"}), accept_at=1.0)
    rule = doc["run"]["acceptance_rule"]
    assert rule["requires"] == []
    assert rule["score"]["name"] == "judge" and rule["score"]["target"] == 1.0
    assert _errors(doc) == []


def test_given_usd_is_kept_with_tokens_and_the_recovered_mark():
    tok = {"input": 1000, "cache_read": 200, "cache_write": 10, "output": 300}
    doc = HR.to_ocp(_row(cost={"usd": 0.42, "tokens": tok, "recovered": True, "usd_source": "opencode"}))
    [att] = doc["attempts"]
    assert att["cost"]["usd"] == 0.42
    assert (att["cost"]["input_tokens"], att["cost"]["cached_input_tokens"], att["cost"]["output_tokens"]) == (1000, 200, 300)
    assert att["cost"]["cache_creation_tokens"] == 10
    assert att["cost"]["basis"] == "measured"
    mark = att["ext"][NS]["cost"]
    assert mark["recovered"] is True and mark["usd_source"] == "opencode"


def test_tokens_without_usd_are_priced_from_the_table():
    tok = {"input": 1000, "cache_read": 200, "cache_write": 0, "output": 300}
    doc = HR.to_ocp(_row(cost={"tokens": tok, "recovered": True}))
    want = price_run({"model": "gpt-5.4", "tokens": {"in": 1000, "cache_read": 200, "cache_write": 0, "out": 300}},
                     load_prices())["usd"]
    [att] = doc["attempts"]
    assert want is not None and att["cost"]["usd"] == pytest.approx(want)
    assert att["ext"][NS]["cost"]["usd_source"] == "loopmath-prices"


def test_an_unpriced_model_keeps_its_tokens_and_the_reason_never_zero():
    tok = {"input": 1000, "cache_read": 0, "cache_write": 0, "output": 300}
    doc = HR.to_ocp(_row(model="glm-9-unlisted", cost={"tokens": tok, "recovered": True}))
    [att] = doc["attempts"]
    assert "usd" not in att["cost"]
    assert att["cost"]["input_tokens"] == 1000
    assert att["ext"][NS]["cost"]["unpriced"]
    assert _errors(doc) == []


def test_no_cost_means_no_cost_numbers():
    [att] = HR.to_ocp(_row(cost=None))["attempts"]
    assert "usd" not in (att.get("cost") or {})
    assert "input_tokens" not in (att.get("cost") or {})


def test_missing_model_and_agent_import_as_unknown():
    doc = HR.to_ocp({"schema": HR.SCHEMA, "task_id": "t", "ended_at": WHEN})
    [setting] = doc["run"]["configuration"]["settings"].values()
    assert setting["harness"] == "unknown" and setting["model"]["id"] == "unknown"
    assert _errors(doc) == []


def test_ids_are_stable_and_distinct():
    a, b = HR.to_ocp(_row()), HR.to_ocp(_row())
    assert a["run"]["id"] == b["run"]["id"] and a == b
    c = HR.to_ocp(_row(tool={"name": "harness2", "run_id": "jb-oc-agg2/ledger-1/base2"}))
    assert c["run"]["id"] != a["run"]["id"]
    d = HR.to_ocp(_row(id="my-row-7"))
    assert d["run"]["id"] == HR.to_ocp(_row(id="my-row-7", model="other"))["run"]["id"]


def test_the_rest_of_the_row_is_kept_in_ext():
    doc = HR.to_ocp(_row(ext={"dev.loopmath.gate": {"replicate": 2}}, stage_notes="kept"))
    kept = doc["run"]["ext"][NS]
    assert kept["row_ext"] == {"dev.loopmath.gate": {"replicate": 2}}
    assert kept["extra"] == {"stage_notes": "kept"}
    assert kept["tool"]["run_id"] == "jb-oc-agg2/ledger-1/base1"
    assert kept["arm"] == "base"


def test_observed_at_falls_back_to_started_then_the_given_default():
    row = _row(ended_at=None, outcome={"score": 1, "judge": "j"})
    assert HR.to_ocp(row)["run"]["signals"][0]["observed_at"] == "2026-10-18T17:10:00-07:00"
    row = _row(ended_at=None, started_at=None, outcome={"score": 1, "judge": "j"})
    doc = HR.to_ocp(row, observed_default="2026-10-17T08:00:00-07:00")
    assert doc["run"]["signals"][0]["observed_at"] == "2026-10-17T08:00:00-07:00"


def test_unknown_execution_times_stay_absent():
    """observed_at may fall back; a run's or attempt's start and end come only from the row."""
    row = _row(started_at=None, ended_at=None, outcome={"score": 1, "judge": "j"})
    doc = HR.to_ocp(row, observed_default="2026-10-17T08:00:00-07:00")
    [att] = doc["attempts"]
    assert "started_at" not in doc["run"] and "started_at" not in att and "ended_at" not in att
    assert not [e for e in doc["events"] if e.get("detail") == "run started"]
    assert doc["run"]["signals"][0]["observed_at"] == "2026-10-17T08:00:00-07:00"
    assert _errors(doc) == [] and _strict_errors(doc) == []


def test_no_time_at_all_never_writes_the_epoch():
    doc = HR.to_ocp(_row(started_at=None, ended_at=None, outcome={"score": 1, "judge": "j"}))
    assert "1970-01-01" not in json.dumps(doc)
    assert "started_at" not in doc["run"] and "started_at" not in doc["attempts"][0]
    assert doc["run"]["signals"][0]["observed_at"]          # when loopmath observed it: now
    assert _strict_errors(doc) == []


def test_an_end_alone_is_the_attempts_end_and_never_its_start():
    doc = HR.to_ocp(_row(started_at=None))
    [att] = doc["attempts"]
    assert att["ended_at"] == WHEN and "started_at" not in att and "started_at" not in doc["run"]


def test_given_times_are_the_runs_and_attempts_times():
    doc = HR.to_ocp(_row())
    [att] = doc["attempts"]
    assert doc["run"]["started_at"] == att["started_at"] == "2026-10-18T17:10:00-07:00"
    assert att["ended_at"] == WHEN


def test_the_pair_key_is_copied_to_its_one_canonical_place():
    """Rows carry `pair`; runs carry dev.loopmath.pair, the one place compare reads a pair key."""
    doc = HR.to_ocp(_row(pair="ledger-1#2", arm="candidate"))
    assert doc["run"]["ext"]["dev.loopmath.pair"] == {"pair": "ledger-1#2", "arm": "candidate", "experiment": None}
    assert _errors(doc) == []
    assert "dev.loopmath.pair" not in HR.to_ocp(_row())["run"]["ext"]


def test_the_experiment_joins_the_pair_object_and_the_ids():
    """A pair is (experiment, pair): the experiment joins the pair object and the row and run ids."""
    doc = HR.to_ocp(_row(experiment="h2-0123456789abcdef"))
    assert doc["run"]["ext"]["dev.loopmath.pair"] == {"pair": None, "arm": "base", "experiment": "h2-0123456789abcdef"}
    assert doc["run"]["ext"][NS]["experiment"] == "h2-0123456789abcdef"
    assert _errors(doc) == []
    assert HR.row_id(_row(experiment="h2-0123456789abcdef")) == "harness2:h2-0123456789abcdef:jb-oc-agg2/ledger-1/base1"
    assert HR.row_id(_row()) == "harness2:jb-oc-agg2/ledger-1/base1"
    ids = {HR.to_ocp(_row(experiment=e))["run"]["id"] for e in (None, "h2-a", "h2-b")}
    assert len(ids) == 3


def test_a_failed_status_is_a_failed_attempt_with_unknown_outcome():
    doc = HR.to_ocp(_row(status="failed"))
    att = doc["attempts"][0]
    assert att["status"] == "failed" and att["outcome"]["result"] == "failed"
    assert doc["run"]["signals"] == [] and "cost" not in att
    assert doc["run"]["ext"][NS]["status"] == "failed"
    assert _errors(doc) == []
    assert HR.to_ocp(_row(status="done"))["attempts"][0]["status"] == "done"


def test_incomplete_counts_never_become_the_attempt_cost():
    tokens = {"input": 900, "cache_read": 100, "cache_write": 0, "output": 80}
    doc = HR.to_ocp(_row(cost={"usd": None, "tokens": tokens, "complete": False, "partial_usd": 0.004,
                               "incomplete": "1 step started with no step_finish after", "recovered": True}))
    att = doc["attempts"][0]
    assert "cost" not in att
    marks = att["ext"][NS]["cost"]
    assert marks == {"recovered": True, "complete": False, "incomplete": "1 step started with no step_finish after",
                     "usd_source": None, "unpriced": "the counts are incomplete", "partial_tokens": tokens,
                     "partial_usd": 0.004}
    assert _errors(doc) == []
    with pytest.raises(ValueError, match="cost.usd: must be null when cost.complete is false"):
        HR.to_ocp(_row(cost={"usd": 0.01, "complete": False}))


def test_a_partial_reported_cost_is_a_note_and_the_tokens_are_priced():
    tokens = {"input": 900, "cache_read": 100, "cache_write": 0, "output": 80}
    doc = HR.to_ocp(_row(cost={"usd": None, "tokens": tokens, "partial_usd": 0.004, "recovered": True}))
    att = doc["attempts"][0]
    priced, _ = HR.price_tokens("gpt-5.4", tokens)
    assert att["cost"]["usd"] == pytest.approx(round(priced, 6))
    assert att["ext"][NS]["cost"]["partial_usd"] == 0.004
    assert att["ext"][NS]["cost"]["usd_source"] == "loopmath-prices"


@pytest.mark.parametrize("name", ["auto-research-loop", "plugin-evals", "hand-ab"])
def test_every_worked_example_imports_as_strict_valid_runs(name):
    examples = Path(__file__).resolve().parents[2] / "spec" / "examples" / "harness-runs"
    rows = HR.read(examples / f"{name}.jsonl")
    docs = [HR.to_ocp(row, base_dir=examples) for row in rows]
    assert len({d["run"]["id"] for d in docs}) == len(rows)
    for doc in docs:
        assert _errors(doc) == [], doc["run"]["id"]
        assert harness.run_record(doc) is not None


def test_free_text_the_row_keeps_makes_the_profile_full():
    """spec/OCP.md section 5: metadata_only promises no prompt or transcript text; a row's own text is kept in ext,
    so the document says full and names where."""
    doc = HR.to_ocp(_row(prompt="Summarise the ledger for the board.", ext={"org.example.loop": {"note": "retry"}},
                         outcome={"score": 0.5, "scale": "fraction", "rationale": "Missed one total."}))
    assert doc["privacy"]["profile"] == "full" and _strict_errors(doc) == []
    assert doc["run"]["ext"][NS]["retained_text"] == ["ext.org.example.loop.note", "outcome.rationale", "prompt"]
    assert doc["run"]["ext"][NS]["extra"]["prompt"] == "Summarise the ledger for the board."


@pytest.mark.parametrize("over", [{}, {"ext": {"org.example.loop": {"editor_usd": 0.035, "rounds": [1, 2]}}},
                                  {"outcome": {"score": 0.5, "scale": "fraction", "n_passed": 2, "n_criteria": 4,
                                               "metric": "weighted", "judge": "judge-x"}},
                                  {"ext": {"dev.loopmath.adapter.harness2": {"bench": "jb", "label": "base1"}}}],
                         ids=["plain", "numbers", "labels", "loopmath-ext"])
def test_labels_and_numbers_keep_metadata_only(over):
    doc = HR.to_ocp(_row(**over))
    assert doc["privacy"]["profile"] == "metadata_only" and "retained_text" not in doc["run"]["ext"][NS]
