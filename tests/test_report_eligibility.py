"""Every analyze surface explains the same comparison population."""
import copy

import pandas as pd
import pytest

from loopmath.cli_support import _extremes_ratio
from loopmath.report.eligibility import unavailable_lines
from loopmath.report.html import _configuration_table
from loopmath.report.terminal import _no_configuration_lines


@pytest.mark.parametrize("counts,word,amount", [
    ({"unknown acceptance": 349, "configuration below min_overlap_cells": 136,
      "configuration below min_n": 9}, "overlap", "136 of 145"),
    ({"configuration below min_n": 12, "configuration below min_overlap_cells": 4}, "minimum run count", "12 of 16"),
    ({"configuration below min_n": 4, "configuration below min_overlap_cells": 4}, "overlap", "4 of 8"),
    ({"run's task-mix cell not shared with another configuration": 7}, "not shared", "7 of 7"),
    ({"unknown acceptance": 5}, "unknown acceptance", "5 of 5"),
    ({"no usable cost value": 5}, "no usable cost", "5 of 5"),
    ({"no model label": 5}, "no model label", "5 of 5"),
])
def test_reason_count_population_and_remedy_agree(counts, word, amount):
    surface = {"table": pd.DataFrame(), "min_n": 5, "n_rows": sum(counts.values()),
               "n_rows_used": 0, "exclusions": [{"reason": r, "n": n} for r, n in counts.items()]}
    original = copy.deepcopy(surface)
    texts = ["\n".join(_no_configuration_lines(surface)), _configuration_table(surface),
             _extremes_ratio(pd.DataFrame(), surface, None)["unavailable"]]
    for text in texts:
        assert word in text and amount in text
        if "overlap" in word or "not shared" in word:
            assert "shared task-mix cells" in text and "lower the minimum" not in text
    assert surface["exclusions"] == original["exclusions"]


def test_zero_counts_and_one_eligible_configuration_make_no_minimum_claim():
    for surface in ({}, {"exclusions": [{"reason": "configuration below min_n", "n": 0}]}):
        assert "minimum" not in " ".join(unavailable_lines(surface))
    result = _extremes_ratio(pd.DataFrame(), {"table": pd.DataFrame([{"arm": "a"}])}, None)
    assert result == {"unavailable": "only one workflow configuration is eligible, so there is nothing to compare"}
