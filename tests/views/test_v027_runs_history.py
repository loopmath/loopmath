"""`loopmath runs` and the runs page count spend as `status` and `budget` do (0.2.7): the runs `onboard` brought in
from the logs are shown apart, with their own total, and are not in the spend. A loop run recorded with
`--source habit` is the user's run and is counted. Rows, outcomes and receipts are unchanged."""

from __future__ import annotations

from loopmath.store import Store
from loopmath.views import runs
from tests.store.store_helpers import finished_doc

# (run, source, dollars, history): None lets the helper decide (habit is onboard's), False is a loop run
RUNS = (("run_u1", "usual", 0.10, None), ("run_e1", "exploration", 0.06, None), ("run_l1", "habit", 0.50, False),
        ("run_h1", "habit", 3.72, None), ("run_h2", "habit", 1.00, None))


def _view(tmp_path) -> dict:
    store = Store(tmp_path / "lm")
    for run, source, usd, history in RUNS:
        store.import_run(finished_doc(run, source=source, usd=usd, history=history))
    return runs.build_view(store.home)


def test_the_terminal_spend_leaves_out_onboard_history_and_names_it(tmp_path):
    data = _view(tmp_path)
    assert sorted(r["run"] for r in data["runs"] if r.get("history")) == ["run_h1", "run_h2"]
    assert all("history" not in r for r in data["runs"] if r["run"] not in ("run_h1", "run_h2"))
    lines = runs.summary_lines(data)
    assert lines[0].startswith("5 runs, $0.66 "), lines[0]
    assert "history from onboard, not counted: $4.72 over 2 runs" in lines


def test_a_store_without_history_reads_as_before(tmp_path):
    store = Store(tmp_path / "lm")
    store.import_run(finished_doc("run_u1", usd=0.10))
    data = runs.build_view(store.home)
    assert "history" not in data["runs"][0]
    assert not [line for line in runs.summary_lines(data) if "history" in line]


def test_the_page_shows_onboard_history_apart(tmp_path, probe):
    page = tmp_path / "runs.html"
    page.write_text(runs.render(_view(tmp_path)), encoding="utf-8")
    out = probe(page, "({lede: document.getElementById('lede').textContent, "
                      "stats: [...document.querySelectorAll('.stat')].map(s => s.innerText.replace(/\\s+/g, ' '))})")
    assert out["exceptions"] == [] and out["console_errors"] == []
    lede, stats = out["result"]["lede"], out["result"]["stats"]
    assert "5 runs" in lede and "cost $0.66 (" in lede, lede
    assert "not counting $4.72 of history from onboard over 2 runs." in lede, lede
    assert any(s.startswith("$0.66 ") and s.endswith("spend") for s in stats), stats
    assert "$4.72 history from onboard, not counted (2 runs)" in stats, stats
