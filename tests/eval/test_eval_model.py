"""The design of a fit: options, the row layout, the identification report. Pure, no engine.

All data here is synthetic: the small study of the test data folder and edits of it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import model as M
from loopmath.eval import quality as Q

STUDY = Path(__file__).parent / "data" / "study-small"


@pytest.fixture(scope="module")
def loaded():
    return (C.read_study(STUDY), C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC),
            C.read_document(STUDY / "judges.json", C.SCHEMA_JUDGES))


@pytest.fixture(scope="module")
def design(loaded):
    return M.build(*loaded)


def state(design, term, judge=None, dimension=None):
    rows = [row for row in design.terms
            if row["term"] == term and row["judge"] == judge and row["dimension"] == dimension]
    assert len(rows) == 1, (term, judge, dimension, rows)
    return rows[0]["state"]


def problem(call) -> str:
    with pytest.raises(C.ContractError) as info:
        call()
    return str(info.value)


# ---------------------------------------------------------------------------------------------------
# Options

def test_terms_default_to_auto_and_refuse_anything_else():
    assert M.parse_terms(None) == {"family": "auto", "judge_input": "auto"}
    assert M.parse_terms(["family=off", "judge_input=on"]) == {"family": "off", "judge_input": "on"}
    assert M.parse_terms(["family=off", "family=on"])["family"] == "on"       # the last one wins
    for bad in ("family", "family=maybe", "halo=on", "=on"):
        with pytest.raises(C.ContractError):
            M.parse_terms([bad])


def test_structure_defaults_follow_the_number_of_dimensions():
    assert M.parse_structure(None, 4) == ("full", 0) and M.parse_structure(None, M.FULL_UP_TO) == ("full", 0)
    assert M.parse_structure(None, M.FULL_UP_TO + 1) == ("factor", 2)
    assert M.parse_structure("diag", 4) == ("diag", 0) and M.parse_structure("factor:3", 4) == ("factor", 3)
    for bad in ("factor", "factor:0", "factor:5", "factor:x", "wide"):
        with pytest.raises(C.ContractError):
            M.parse_structure(bad, 4)


def test_what_is_not_built_is_refused_by_name(loaded):
    assert "the full halo form is not built yet" in problem(lambda: M.build(*loaded, halo="full"))
    assert "free cut points are not built yet" in problem(lambda: M.build(*loaded, cuts="free"))
    assert 'unknown component "nope"' in problem(lambda: M.build(*loaded, components=["nope"]))
    study, rubric, judges = loaded
    assert "the study has no runs" in problem(lambda: M.build(C.Study(study.folder), rubric, judges))
    bare = C.Study(study.folder, study.runs)
    assert "no reading and no pair can be fitted" in problem(lambda: M.build(bare, rubric, judges))
    fake = C.Study(study.folder, [dict(study.runs[0], simulated=True)] + study.runs[1:], study.scores)
    assert "the run is simulated" in problem(lambda: M.build(fake, rubric, judges))
    assert M.build(fake, rubric, judges, allow_simulated=True).counts["readings_used"] == 304


# ---------------------------------------------------------------------------------------------------
# The layout

def test_every_used_reading_is_laid_out_once(design):
    data = design.data
    assert design.sizes == {"A": 2, "D": 4, "J": 3, "F": 2, "I": 8, "R": 16}
    assert design.counts == {"readings": 305, "readings_used": 304, "pairs": 18, "pairs_used": 18}
    assert design.notes == [{"kind": "reading", "id": "s0021", "reason": "replaced by s0305"}]
    assert int(data["s_n"].sum()) + len(data["g_r"]) + len(data["k_r"]) == 304
    assert len(data["p_v"]) == 18 and set(data["p_v"]) <= {0, 1, 2} and set(data["p_o"]) <= {-1.0, 0.0, 1.0}
    assert sum(int(group.size) for group in data["k_groups"]) == len(data["k_r"])
    for group in data["k_groups"]:             # one row per cell: one judge, one run, one dimension
        for name in ("k_j", "k_r", "k_d"):
            assert np.all(data[name][group] == data[name][group][:, :1])
    assert data["structure"] == "full" and design.options["structure"] == "full"
    assert design.options["terms"] == {"family": "auto", "judge_input": "auto"}
    for name in ("s_r", "g_r", "k_r", "p_ra", "p_rb"):
        assert np.all((data[name] >= 0) & (data[name] < 16)), name
    for name in ("g_cell", "p_ca", "p_cb"):
        assert np.all(data[name] < data["n_cell"]), name
    assert np.all(data["p_ca"] >= 0) and np.all(data["p_cb"] >= 0)   # a pair always has its two cells
    assert data["n_pc"] == 0 and data["n_e"] == 0                    # no panel judge, no items


def test_linear_cells_go_in_as_sufficient_statistics(design):
    data = design.data
    kinds = [dim["scale"]["kind"] for dim in design.dims]
    assert kinds == ["binary", "levels", "linear", "linear"]
    assert set(data["s_d"]) == {2, 3} and set(data["g_kind"]) <= {M.BINARY, M.LEVELS}
    assert np.all((data["s_ybar"] >= 0) & (data["s_ybar"] <= 1)) and np.all(data["s_ss"] >= 0)
    assert np.all(data["s_ss"][data["s_n"] == 1] == 0)
    # a cell that a pair looks at is linked to its statistics, so its part is drawn given the readings
    linked = data["cell_suff"] >= 0
    assert linked.any() and np.all(data["cell_d"][linked] == 3)
    assert np.all(data["s_j"][data["cell_suff"][linked]] == data["cell_j"][linked])


def test_binary_and_levels_readings_are_intervals_at_the_fixed_cuts(design):
    data = design.data
    cuts = Q.fixed_cuts(3)
    for prefix in ("g", "k"):
        lo, hi, d = data[f"{prefix}_lo"], data[f"{prefix}_hi"], data[f"{prefix}_d"]
        assert np.all(lo < hi)
        binary = d == 0
        assert set(zip(lo[binary], hi[binary])) <= {(0.0, M.OPEN), (-M.OPEN, 0.0)}
        levels = d == 1
        allowed = {(-M.OPEN, float(cuts[0])), (float(cuts[0]), float(cuts[1])), (float(cuts[1]), M.OPEN)}
        assert set(zip(lo[levels], hi[levels])) <= allowed
    assert list(cuts) == [0.25, 0.75]                                # halfway between the levels
    assert np.all(np.isin(data["k_d"], [0, 1]))                      # only cut readings repeat into groups


def test_units_one_anchor_carries_the_bias_and_the_binary_unit(design):
    data = design.data
    assert np.allclose(data["bias_w"].sum(axis=0), 1.0) and np.all(data["bias_w"][0] == 1.0)
    assert list(data["sigma_fixed"]) == [True, False, False, False]
    assert data["sigma_w"][0, 0] == 1.0 and state(design, "sigma", "reviewer", "final/numbers_right") == "off"
    assert list(data["scale"]) == [M.PROBIT_SCALE, M.UNIT_SCALE, M.UNIT_SCALE, M.UNIT_SCALE]
    assert list(data["mu_loc"]) == [0.0, 0.5, 0.5, 0.5]
    assert data["person_tight"].tolist() == [[False, False, True, True], [False] * 4, [False] * 4]
    assert list(data["levels_dim"]) == [False, True, False, False]


def test_without_an_anchor_the_unit_is_the_mean_of_the_judges(loaded):
    study, rubric, judges = loaded
    free = copy.deepcopy(judges)
    free["anchor"] = []
    design = M.build(study, rubric, free)
    assert design.anchor["kind"] == "judges_mean" and design.anchor["judges"] == []
    assert {row["state"] for row in design.anchor["dimensions"]} == {"relative"}
    assert np.allclose(design.data["bias_w"], 1.0 / 3.0) and np.allclose(design.data["sigma_w"], 1.0 / 3.0)
    assert state(design, "sigma", "reviewer", "final/numbers_right") != "off"   # no single judge is the unit
    assert {row["state"] for row in design.terms if row["term"] == "anchor_bias"} == {"prior_only"}


def test_too_few_anchored_targets_is_relative_and_enough_is_weak(loaded):
    study, rubric, judges = loaded
    strict = dict(copy.deepcopy(judges), min_anchor_targets=11)
    assert {row["state"] for row in M.build(study, rubric, strict).anchor["dimensions"]} == {"relative"}
    design = M.build(study, rubric, judges)
    assert [(row["state"], row["n_targets"]) for row in design.anchor["dimensions"]] == [("weak", 10)] * 4
    assert design.anchor["kind"] == "person" and design.anchor["min_anchor_targets"] == 6


def test_a_dimension_with_pairs_only_is_relative(loaded):
    study, rubric, judges = loaded
    scores = [row for row in study.scores if row["component"] != "overall"]
    design = M.build(C.Study(study.folder, study.runs, scores, study.pairs), rubric, judges)
    assert design.relative == ["final/overall"] and list(design.data["pairs_only"]) == [False, False, False, True]
    assert design.data["scale"][3] == M.PROBIT_SCALE and design.data["mu_loc"][3] == 0.0
    assert state(design, "sigma", "judge-strong", "final/overall") == "absent"
    assert state(design, "tau", "judge-strong", "final/overall") == "prior_only"
    assert state(design, "tau", "judge-cheap", "final/overall") == "absent"
    assert state(design, "anchor_bias", None, "final/overall") == "absent"
    assert design.anchor["dimensions"][3]["state"] == "relative"
    assert "relative (pairs only, no gate can use it): final/overall" in M.report_lines(design)
    assert np.all(design.data["bias_w"][:, 3] == 0.0) and design.data["pair_w"][0, 3] == 1.0


def test_components_narrow_the_fit_and_note_the_rest(loaded):
    design = M.build(*loaded, components=["depth"], structure="diag")
    assert [dim["id"] for dim in design.dims] == ["final/depth"] and design.options["components"] == ["depth"]
    assert design.options["structure"] == "diag" and design.counts["pairs_used"] == 0
    reasons = {row["reason"] for row in design.notes}
    assert "its step and component are not a fitted dimension" in reasons
    assert len(design.notes) == 305 + 18 - design.counts["readings_used"]
    assert not design.data["halo_on"].any()                          # one dimension per target: no halo
    assert {row["state"] for row in design.terms if row["term"] == "halo_split"} == {"absent"}


def test_rows_that_cannot_be_read_become_notes(loaded):
    study, rubric, judges = loaded
    scores = [dict(row) for row in study.scores]
    scores[0]["judge"], scores[1]["run"], scores[2]["rubric"] = "nobody", "no-run", "other@9"
    pairs = [dict(row) for row in study.pairs]
    pairs[0]["b"] = next(run["run"] for run in study.runs if run["input"] != study.runs[0]["input"]
                         and run["run"] != pairs[0]["a"]
                         and run["input"] != next(r["input"] for r in study.runs if r["run"] == pairs[0]["a"]))
    design = M.build(C.Study(study.folder, study.runs, scores, pairs), rubric, judges)
    by_id = {row["id"]: row["reason"] for row in design.notes}
    assert by_id[scores[0]["id"]] == 'unknown judge "nobody"' and by_id[scores[1]["id"]] == 'unknown run "no-run"'
    assert by_id[scores[2]["id"]].startswith("scored under rubric other@9, not ")
    assert by_id[pairs[0]["id"]] == "its two runs have different inputs"
    assert design.counts["pairs_used"] == 17


# ---------------------------------------------------------------------------------------------------
# The identification report

def test_every_term_has_one_row_a_known_state_and_absent_rows_say_why(design):
    keys = [(row["term"], row["judge"], row["dimension"]) for row in design.terms]
    assert len(keys) == len(set(keys))
    assert {row["term"] for row in design.terms} == set(F.TERMS)
    assert {row["state"] for row in design.terms} <= set(F.TERM_STATES)
    assert all(row.get("reason") for row in design.terms if row["state"] == "absent")
    for judge in design.index["judges"]:
        for dim in design.dims:
            assert state(design, "tau", judge, dim["id"]) and state(design, "sigma", judge, dim["id"])


def test_optional_terms_follow_the_design_and_the_switches(loaded, design):
    assert list(design.data["family_on"]) == [False, True, True] and not design.data["ji_on"].any()
    assert state(design, "family", "reviewer") == "absent" and state(design, "family", "judge-strong") == "identified"
    assert state(design, "judge_input", "judge-strong") == "off"     # 8 inputs: under the 10 the term needs
    off = M.build(*loaded, terms=["family=off"])
    assert not off.data["family_on"].any() and state(off, "family", "judge-strong") == "off"
    on = M.build(*loaded, terms=["judge_input=on"])
    assert on.data["ji_on"].all() and on.data["n_ji"] > 0 and state(on, "judge_input", "judge-strong") == "weak"
    assert np.all(on.data["ji_j"] < 3) and np.all(on.data["ji_d"] < 4)


def test_pair_terms_follow_what_the_pairs_can_show(design):
    assert state(design, "position", "judge-cheap") == "absent"
    assert state(design, "position", "reviewer") == "prior_only"     # one order only, never both
    assert state(design, "position", "judge-strong") in ("weak", "identified")
    assert state(design, "sigma_pair", "judge-cheap") == "absent"


def test_the_report_lists_counts_and_what_a_weak_term_needs(design):
    lines = M.report_lines(design)
    assert lines[0] == "design: 16 runs of 2 arms on 8 inputs, 4 dimensions, 3 judges"
    assert lines[2] == "anchor: person" and "terms (identified, weak, prior_only, confounded, off):" in lines
    assert not any(" absent " in line for line in lines)
    assert all("(needs: " in line for line in lines if " prior_only " in line or " weak  " in line)
    obj = M.report(design)
    assert obj["counts"]["not_used"] == 1 and obj["terms"] is design.terms and obj["relative"] == []
