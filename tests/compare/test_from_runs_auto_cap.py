"""`check-change from-runs` picks its cap from the pairs the runs already hold.

The runs exist, so no cap is buying pairs: without `--max-pairs` the cap is the smallest calibrated cap above
the pairs recorded (6, 12, 24 or 48), so a check that has not decided says `more`, not `the base stays`.
Past 48 pairs the cap keeps doubling and no false adopt bound is stated. `--max-pairs` still sets the cap.
JSON and text both say which cap was used and why.
"""

from __future__ import annotations

import pytest

from tests.compare.test_check_change_cli import ARMS, home, import_rows, lm  # noqa: F401  (home: fixture)
from tests.compare.test_compare_paired import verdict_rows

UNDECIDED = ((True, False), (True, True, False))  # P(better) about 0.82 at 24 pairs: between reject and adopt


@pytest.mark.parametrize(("n", "cap"), [(3, 6), (6, 12), (24, 48), (30, 48)])
def test_the_cap_is_the_smallest_calibrated_cap_above_the_pairs(home, tmp_path, capsys, n, cap):
    import_rows(capsys, tmp_path, *verdict_rows(n, UNDECIDED))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0, err
    assert out["cap"] == {"max_pairs": cap, "from": "auto", "pairs": n, "why": out["cap"]["why"]}
    assert f"smallest calibrated cap above the {n} pairs" in out["cap"]["why"]
    assert out["comparison"]["max_pairs"] == cap and out["comparison"]["bound"] is not None
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS)
    assert code == 0, err
    line = text.splitlines()[-1]
    assert line.startswith(f"cap: {cap} pairs, chosen for the {n} pairs recorded"), text
    assert "--max-pairs" in line


def test_at_a_calibrated_cap_an_undecided_check_says_more_not_the_base_stays(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(24, UNDECIDED))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0, err
    cmp = out["comparison"]
    assert (cmp["verdict"], cmp["stop"], cmp["max_pairs"]) == ("more", None, 48), cmp
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--max-pairs", "24", "--json")
    assert (out["comparison"]["verdict"], out["comparison"]["stop"]) == ("reject", "cap"), err
    assert out["cap"] == {"max_pairs": 24, "from": "--max-pairs", "pairs": 24, "why": out["cap"]["why"]}
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS, "--max-pairs", "24")
    assert text.splitlines()[-1].startswith("cap: 24 pairs, from --max-pairs"), text


def test_past_the_calibrated_caps_the_cap_doubles_and_no_bound_is_stated(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(50, ((True, False), (True, False, True))))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0, err
    assert out["cap"]["max_pairs"] == 96 and out["cap"]["from"] == "auto" and out["cap"]["pairs"] == 50
    assert "past the calibrated caps" in out["cap"]["why"] and "no false adopt bound" in out["cap"]["why"]
    cmp = out["comparison"]
    assert cmp["bound"] is None and (cmp["verdict"], cmp["stop"]) == ("more", None), cmp
