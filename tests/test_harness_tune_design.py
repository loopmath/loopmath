"""The probe designs: the PB12 array and its column choice, one-at-a-time, full, fold-over, repeats, df, aliasing."""

from __future__ import annotations

import itertools

import pytest

from loopmath.harness.tune import design as D


def ids(m):
    return [f"ed.e{i}" for i in range(1, m + 1)]


def test_pb12_is_the_cyclic_generator_plus_a_row_of_minus_ones():
    a = D.pb12()
    gen = [1 if c == "+" else -1 for c in D.PB12_GENERATOR]
    assert len(a) == 12 and all(len(r) == 11 for r in a)
    shifts = [gen[-k:] + gen[:-k] if k else gen for k in range(11)]
    assert sorted(map(tuple, a[:11])) == sorted(map(tuple, shifts))
    assert a[11] == [-1] * 11
    for i, j in itertools.combinations(range(11), 2):
        assert sum(r[i] * r[j] for r in a) == 0


@pytest.mark.parametrize("m", range(1, 11))
def test_pb12_columns_stay_balanced_and_orthogonal(m):
    p = D.plan_design(ids(m), seed=7).properties()
    assert p["runs"] == 12 and p["on_per_edit"] == [6] and p["orthogonal"]
    if m >= 5:
        assert p["all_off_runs"] == 0 and p["all_on_runs"] == 0


def test_pb12_flips_spread_the_edits_per_run():
    p = D.plan_design(ids(10), seed=0).properties()
    sizes = p["edits_on_per_run"]
    assert min(sizes) >= 1 and max(sizes) <= 9
    assert sum((2 * s - 10) ** 2 for s in sizes) <= sum((2 * s - 10) ** 2 for s in
                                                        [sum(1 for v in r[:10] if v > 0) for r in D.pb12()])


def test_the_seed_sets_columns_and_order_and_repeats_exactly():
    a, b, c = D.plan_design(ids(6), seed=1), D.plan_design(ids(6), seed=1), D.plan_design(ids(6), seed=2)
    assert a.rows == b.rows and a.columns == b.columns
    assert (a.rows, a.columns) != (c.rows, c.columns)
    assert sorted(a.columns) == sorted(set(a.columns)) and all(1 <= x <= 11 for x in a.columns)
    assert D.seed_of("task", 0, 1) == D.seed_of("task", 0, 1) != D.seed_of("task", 0, 2)


def test_one_at_a_time_full_and_fold_over():
    one = D.plan_design(ids(3), "one-at-a-time", seed=0)
    assert sorted(map(tuple, one.rows)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    full = D.plan_design(ids(4), "full", seed=0)
    assert full.n == 16 and sorted(map(tuple, full.rows)) == sorted(itertools.product((0, 1), repeat=4))
    fo = D.plan_design(ids(3), "fold-over", seed=1, pairs=4)
    assert fo.n == 8 and fo.pair == [1, 1, 2, 2, 3, 3, 4, 4]
    for r in range(0, 8, 2):
        assert [1 - v for v in fo.rows[r]] == fo.rows[r + 1]


def test_repeats_copy_the_rows_and_number_them():
    d = D.plan_design(ids(2), "one-at-a-time", seed=0, repeats=2)
    assert d.rows[:2] == d.rows[2:] and d.repeat == [1, 1, 2, 2] and d.row_of == [0, 1, 0, 1]
    p = D.plan_design(ids(10), seed=0, repeats=2)
    assert p.n == 24 and p.row_of == list(range(12)) * 2


def test_on_and_signed():
    d = D.plan_design(ids(3), "one-at-a-time", seed=0)
    for r in range(d.n):
        assert len(d.on(r)) == 1 and d.signed()[r].count(1) == 1 and d.signed()[r].count(-1) == 2


@pytest.mark.parametrize("args,match", [
    ((ids(11), "pb12"), "at most 10"),
    ((ids(5), "full"), "at most 4"),
    ((["a", "a"], "pb12"), "distinct"),
    ((ids(2), "latin"), "unknown design"),
])
def test_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        D.plan_design(*args)
    with pytest.raises(ValueError, match="repeats"):
        D.plan_design(ids(2), repeats=0)


def test_an_empty_design_has_no_rows():
    d = D.plan_design([], seed=0)
    assert d.n == 0 and d.m == 0 and d.properties() == {"runs": 0, "edits": 0}


def test_probe_count_and_residual_df():
    assert D.probe_count("pb12", 10) == 12 and D.probe_count("pb12", 0) == 0
    assert D.probe_count("fold-over", 3, pairs=8) == 16 and D.probe_count("full", 3) == 8
    assert D.probe_count("one-at-a-time", 4, repeats=3) == 12
    assert D.residual_df("pb12", 10) == 3
    assert D.residual_df("pb12", 10, base_rows=False) == 1
    assert D.residual_df("pb12", 4) == 9
    assert D.residual_df("pb12", 0) is None


def test_alias_summary_of_the_default_design():
    a = D.alias_summary(D.plan_design(ids(10), seed=0))
    assert a["design_only"] == {"max_leak_outside_pair": pytest.approx(1 / 3, abs=1e-4),
                                "max_leak_inside_pair": 0.0, "se_factor": [0.5774, 0.5774]}
    wb = a["with_base"]
    assert wb["max_leak_outside_pair"] == pytest.approx(0.4706, abs=1e-4)
    assert wb["max_leak_inside_pair"] == pytest.approx(0.1373, abs=1e-4)
    assert wb["se_factor"] == [0.5601, 0.5601]
    assert D.alias_summary(D.plan_design(ids(1), seed=0)) == {"with_base": None, "design_only": None}


def test_a_design_record_is_json_ready():
    import json

    d = D.plan_design(ids(4), seed=5)
    back = D.Design(**json.loads(json.dumps(d.to_dict())))
    assert back == d
