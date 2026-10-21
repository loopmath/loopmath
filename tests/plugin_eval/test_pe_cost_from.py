"""`--cost-from FILE`: a user's own per run dollars or tokens joined to a plugin eval result by case, arm and run.
The result itself carries only the suite's cost (https://code.claude.com/docs/en/plugin-evals#json-result), so
without the file every run's cost is unknown, never 0 and never a share of the suite's."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pefixture as F  # noqa: E402

from loopmath import harness_runs as HR  # noqa: E402
from loopmath.adapters import plugin_eval as PE  # noqa: E402

TOKENS = {"input": 1000, "cache_read": 5000, "cache_write": 0, "output": 200}


@pytest.fixture
def suite(tmp_path) -> F.Built:
    return F.write(tmp_path / "p", F.docs_suite())


def _row(rows, case, arm, j):
    return next(r for r in rows if r["task_id"] == case and r["arm"] == arm and r["pair"] == f"{case}#{j}")


def test_dollar_lines_become_each_runs_reported_cost(suite, tmp_path):
    costs = F.write_costs(tmp_path / "c.jsonl", [{"case": "fix-null-check", "arm": "with", "run": 2, "usd": 0.41},
                                                 {"case": "fix-null-check", "arm": "without", "run": 2, "usd": 0.5}])
    got = PE.read(suite.path, cost_from=costs)
    assert _row(got.rows, "fix-null-check", "with", 2)["cost"] == {"usd": 0.41, "usd_source": "cost-from"}
    assert _row(got.rows, "fix-null-check", "without", 2)["cost"]["usd"] == 0.5
    assert _row(got.rows, "fix-null-check", "with", 1).get("cost") is None
    assert got.summary["cost_from"] == {"lines": 2, "runs_with_cost": 2, "runs_without": 22}


def test_token_lines_are_priced_at_import_and_an_unknown_model_stays_unpriced(suite, tmp_path):
    costs = F.write_costs(tmp_path / "c.jsonl", [
        {"case": "write-docs-page", "arm": "with", "run": 1, "tokens": TOKENS, "model": "claude-sonnet-5"},
        {"case": "write-docs-page", "arm": "with", "run": 2, "tokens": TOKENS, "model": "claude-nonesuch-9"}])
    got = PE.read(suite.path, cost_from=costs)
    priced = _row(got.rows, "write-docs-page", "with", 1)
    assert priced["cost"]["tokens"] == TOKENS and priced["cost"].get("usd") is None and priced["model"] == "claude-sonnet-5"
    doc = HR.to_ocp(priced)
    assert doc["attempts"][0]["cost"]["usd"] > 0   # priced from loopmath's table
    unknown = HR.to_ocp(_row(got.rows, "write-docs-page", "with", 2))
    assert (doc_cost := unknown["attempts"][0].get("cost")) is None or doc_cost.get("usd") is None   # never 0


@pytest.mark.parametrize("line, why", [
    ({"case": "no-such-case", "arm": "with", "run": 1, "usd": 0.1}, "no-such-case"),
    ({"case": "fix-null-check", "arm": "both", "run": 1, "usd": 0.1}, "arm"),
    ({"case": "fix-null-check", "arm": "with", "run": 4, "usd": 0.1}, "run 4"),
    ({"case": "fix-null-check", "arm": "with", "run": 0, "usd": 0.1}, "run"),
    ({"case": "fix-null-check", "arm": "with", "run": 1}, "usd or tokens"),
    ({"case": "fix-null-check", "arm": "with", "run": 1, "usd": -1}, "usd"),
])
def test_a_line_that_matches_no_run_or_says_nothing_is_refused_by_line(suite, tmp_path, line, why):
    costs = F.write_costs(tmp_path / "c.jsonl", [{"case": "fix-null-check", "arm": "with", "run": 1, "usd": 0.2}, line])
    with pytest.raises(PE.FormatError, match=rf"c\.jsonl:2: .*{why}"):
        PE.read(suite.path, cost_from=costs)


def test_two_lines_for_one_run_are_refused(suite, tmp_path):
    line = {"case": "fix-null-check", "arm": "with", "run": 1, "usd": 0.2}
    costs = F.write_costs(tmp_path / "c.jsonl", [line, {**line, "usd": 0.3}])
    with pytest.raises(PE.FormatError, match=r"c\.jsonl:2: .*already"):
        PE.read(suite.path, cost_from=costs)


def test_the_twin_cost_file_gives_the_shipped_rows_dollars(tmp_path):
    built = F.write(tmp_path / "twin", F.twin_suite())
    costs = F.write_costs(tmp_path / "twin-costs.jsonl", F.TWIN_COSTS)
    rows = PE.read(built.path, cost_from=costs, model="claude-sonnet-5-5").rows
    assert sorted(r["cost"]["usd"] for r in rows) == [0.27, 0.28, 0.29, 0.31, 0.32, 0.33]
