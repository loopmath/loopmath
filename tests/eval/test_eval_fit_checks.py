"""`eval fit`: saved draws, call uniqueness, empty dimensions, what convergence covers.

No test here runs the engine. Each compares with a value worked out another way (a known sd, a split
R-hat written out from its definition, the report of `eval scores check`) or pins a refusal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import fit as FT
from loopmath.eval import model as M
from loopmath.eval import scores as SC

STUDY = Path(__file__).parent / "data" / "study-small"
RUBRIC, JUDGES = str(STUDY / "rubric.json"), str(STUDY / "judges.json")
BEFORE = tuple(name for name in FT.CHECKED if name not in FT.LATER)


@pytest.fixture(scope="module")
def loaded():
    return (C.read_study(STUDY), C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC),
            C.read_document(STUDY / "judges.json", C.SCHEMA_JUDGES))


def refusal(call) -> C.ContractError:
    with pytest.raises(C.ContractError) as info:
        call()
    return info.value


def plain_split_rhat(x: np.ndarray) -> float:
    """Split R-hat of `x[chains, draws]` from its definition, written apart from `fit.split_rhat_ess`."""
    half = x.shape[1] // 2
    parts = [chain[:half] for chain in x] + [chain[half:2 * half] for chain in x]
    within = float(np.mean([np.var(part, ddof=1) for part in parts]))
    between = half * float(np.var([np.mean(part) for part in parts], ddof=1))
    return float(np.sqrt(((half - 1) / half * within + between / half) / within))


def chains(rng, shape=(2, 400)) -> dict:
    """Well mixed draws of every checked array: 2 dimensions, 600 runs."""
    raw = {name: rng.standard_normal(shape + (2,)) for name in BEFORE}
    for name in ("corr_input", "corr_run"):
        corr = np.broadcast_to(np.eye(2), shape + (2, 2)).copy()
        corr[..., 0, 1] = corr[..., 1, 0] = 0.1 * rng.standard_normal(shape)
        raw[name] = corr
    raw["q_run"] = rng.standard_normal(shape + (600, 2))
    return raw


# ---------------------------------------------------------------------------------------------------
# The saved draws carry the reported uncertainty, and the diagnostics describe them

def test_keep_draws_below_the_minimum_is_refused_before_anything_is_sampled(loaded, monkeypatch, tmp_path, capsys):
    design = M.build(*loaded)
    monkeypatch.setattr(engine, "sample", lambda *args, **kwargs: pytest.fail("the engine ran"))
    for keep in (1, FT.MIN_KEEP_DRAWS - 1):
        found = refusal(lambda: FT.fit_design(design, keep_draws=keep))
        assert found.code == "bad_option" and found.where == "--keep-draws"
        assert str(found) == (f"eval fit: --keep-draws: {keep} is below 400: the reported 90% intervals are read "
                              "from the saved draws and need that many; give 400 or more (the default is 1000)")
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--out", str(out),
                     "--keep-draws", "1"]) == 1
    assert capsys.readouterr().err.startswith("error: eval fit: --keep-draws: 1 is below 400")
    assert list(tmp_path.iterdir()) == []


def test_thinning_keeps_every_chain_and_the_spread_of_the_draws():
    rng = np.random.default_rng(11)
    raw = {name: 0.3 + 0.02 * rng.standard_normal((4, 1000, 2)) for name in FT.CHECKED}
    kept = FT.thin(raw, 400)
    assert {x.shape for x in kept.values()} == {(4, 100, 2)}
    picks = np.round(np.linspace(0, 999, 100)).astype(int)
    assert np.array_equal(kept["mu"], raw["mu"][:, picks])
    # the known sd of the draws is 0.02: 400 saved draws give it within 3 standard errors (0.02 / sqrt(800))
    assert abs(kept["mu"][:, :, 0].std(ddof=1) - 0.02) < 3 * 0.02 / np.sqrt(800)
    assert FT.thin(raw, 4000)["mu"].shape == (4, 1000, 2) and FT.thin(raw, 5000)["mu"].shape == (4, 1000, 2)
    assert FT.thin(raw, 1001)["mu"].shape == (4, 250, 2)           # never more than asked for


def test_the_diagnostics_describe_the_saved_draws_not_the_chains_before_thinning():
    rng = np.random.default_rng(12)
    raw = {name: 0.3 + 0.02 * rng.standard_normal((4, 1000, 2)) for name in FT.CHECKED}
    kept = FT.thin(raw, 400)
    told = FT.diagnose(kept, {}, 40, 0.0, sampled=4000, least=FT.MIN_KEEP_DRAWS)
    assert [check["name"] for check in told["checks"]] == ["rhat", "ess", "divergences", "saved_draws", "finite_draws"]
    assert told["converged"] and told["checks"][3] == {"name": "saved_draws", "value": 400, "limit": 400, "ok": True}
    assert told["checks"][2]["limit"] == 40.0                        # 1% of the 4000 draws sampled
    assert not FT.diagnose(kept, {}, 41, 0.0, sampled=4000, least=FT.MIN_KEEP_DRAWS)["converged"]
    # 400 saved draws cannot be reported as more effective draws than 400 of them can be
    assert told["ess_min"] <= 400 * np.log10(400) and told["ess_min"] < FT.diagnose(raw, {}, 0, 0.0)["ess_min"]
    # a fit of too few draws to save 400 is written, and is not converged
    few = FT.thin({name: x[:2, :150] for name, x in raw.items()}, 400)
    short = FT.diagnose(few, {}, 0, 0.0, sampled=300, least=FT.MIN_KEEP_DRAWS)
    assert few["mu"].shape == (2, 150, 2) and not short["converged"]
    assert [check["ok"] for check in short["checks"]] == [True, True, True, False, True]
    assert short["checks"][3]["value"] == 300


# ---------------------------------------------------------------------------------------------------
# Two rows of one call refuse, as in `eval scores check`

def test_a_call_with_two_readings_or_two_pairs_refuses_like_scores_check(loaded):
    study, rubric, judges = loaded
    first = next(row for row in study.scores if "item" not in row and row.get("replaces") is None
                 and row["id"] != "s0021")
    twice = C.Study(study.folder, study.runs, [*study.scores, {**first, "id": "s9001"}], study.pairs)
    found = refusal(lambda: M.build(twice, rubric, judges))
    checked = [row for row in SC.check_study(twice, rubric, judges)["problems"] if row["code"] == "call_twice"]
    assert found.code == "call_twice" and found.file == C.SCORES_FILE and len(checked) == 1
    assert str(found) == checked[0]["message"] and f'"{first["id"]}" and "s9001"' in str(found)
    # a repeat with a call of its own is one more reading
    base = M.build(study, rubric, judges)
    repeat = C.Study(study.folder, study.runs, [*study.scores, {**first, "id": "s9001", "call": "call-9001"}],
                     study.pairs)
    assert M.build(repeat, rubric, judges).counts["readings_used"] == base.counts["readings_used"] + 1
    pair = study.pairs[0]
    pairs_twice = C.Study(study.folder, study.runs, study.scores, [*study.pairs, {**pair, "id": "p9001"}])
    found = refusal(lambda: M.build(pairs_twice, rubric, judges))
    checked = [row for row in SC.check_study(pairs_twice, rubric, judges)["problems"] if row["code"] == "call_twice"]
    assert found.code == "call_twice" and found.file == C.PAIRS_FILE and str(found) == checked[0]["message"]


# ---------------------------------------------------------------------------------------------------
# A dimension nothing was read on is not fitted from its prior

def test_a_dimension_with_no_reading_and_no_pair_refuses_and_points_to_components(loaded):
    study, rubric, judges = loaded
    overall = C.overall_component(rubric, "final")
    scores = [row for row in study.scores if row["component"] != "depth"]
    pairs = [row for row in study.pairs if (row["component"] or overall) != "depth"]
    bare = C.Study(study.folder, study.runs, scores, pairs)
    found = refusal(lambda: M.build(bare, rubric, judges))
    rest = [dim["component"] for dim in C.dimensions(rubric) if dim["component"] != "depth"]
    assert found.code == "empty_dimension" and found.where == 'dimension "final/depth"'
    assert found.fix == "score it, or fit the components that have rows: --components " + " ".join(rest)
    assert "from the prior alone" in found.problem
    design = M.build(bare, rubric, judges, components=rest)       # the fix it names works
    assert [dim["component"] for dim in design.dims] == rest
    assert refusal(lambda: M.build(bare, rubric, judges, components=["depth", rest[0]])).code == "empty_dimension"


# ---------------------------------------------------------------------------------------------------
# Convergence covers the correlations and the target qualities

def test_convergence_reads_the_correlations_and_the_target_qualities():
    rng = np.random.default_rng(13)
    raw = chains(rng)
    assert {"corr_input", "corr_run", "q_run"} <= set(FT.CHECKED)
    good = FT.diagnose(raw, {}, 0, 0.0)
    assert good["converged"] and good["rhat_max"] < FT.RHAT_MAX      # the constant diagonal is not checked
    for name in ("corr_input", "corr_run"):
        stuck = raw[name].copy()                                     # two chains near opposite correlations
        stuck[0, :, 0, 1] = stuck[0, :, 1, 0] = 0.8 + 0.01 * rng.standard_normal(400)
        stuck[1, :, 0, 1] = stuck[1, :, 1, 0] = -0.8 + 0.01 * rng.standard_normal(400)
        told = FT.diagnose(dict(raw, **{name: stuck}), {}, 0, 0.0)
        assert not told["converged"] and told["checks"][0]["worst"] in (f"{name}[0, 1]", f"{name}[1, 0]")
        assert told["rhat_max"] == pytest.approx(plain_split_rhat(stuck[:, :, 0, 1]), rel=1e-9)
        assert told["rhat_max"] > 50
    apart = raw["q_run"].copy()                                      # one target, in a later block of columns
    apart[1, :, 500, 1] += 6.0
    told = FT.diagnose(dict(raw, q_run=apart), {}, 0, 0.0)
    assert 500 * 2 + 1 > FT.BLOCK and not told["converged"] and told["checks"][0]["worst"] == "q_run[500, 1]"
    assert told["rhat_max"] == pytest.approx(plain_split_rhat(apart[:, :, 500, 1]), rel=1e-9)
    slow = raw["q_run"].copy()                                       # a target that barely moves: too few draws
    slow[:, :, 7, 0] = np.cumsum(0.05 * rng.standard_normal((2, 400)), axis=1)
    told = FT.diagnose(dict(raw, q_run=slow), {}, 0, 0.0)
    assert not told["checks"][1]["ok"] and told["checks"][1]["worst"] == "q_run[7, 0]"


def test_the_blocks_give_the_same_numbers_as_one_pass(monkeypatch):
    raw = chains(np.random.default_rng(14))
    blocked = FT.diagnose(raw, {}, 0, 0.0)
    monkeypatch.setattr(FT, "BLOCK", 10 ** 6)
    whole = FT.diagnose(raw, {}, 0, 0.0)
    assert blocked["rhat_max"] == pytest.approx(whole["rhat_max"], rel=1e-12)
    assert blocked["ess_min"] == pytest.approx(whole["ess_min"], rel=1e-9)
    assert blocked["checks"][0]["worst"] == whole["checks"][0]["worst"]
