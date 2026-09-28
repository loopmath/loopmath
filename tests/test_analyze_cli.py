"""`loopmath analyze` for a new user: why the table is empty, no band for an n/a spread, `--json` and
`--home` (new-user test P2-18, P3-25)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd

from loopmath.cli import main
from loopmath.cli_support import analyze_payload
from loopmath.report.terminal import beat2_configurations, render
from loopmath.surface_presentation import _band_support_note, shown_band_support_note, walkdown_line

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _surface(**over):
    s = {"ci": 0.8, "cost_col": "usd", "S_naive": math.nan, "S_matched": math.nan, "S_honest": math.nan,
         "S_lo": 1.3, "S_hi": 4.0, "table": pd.DataFrame([{"arm": "a"}])}
    s.update(over)
    return s


def test_no_band_for_an_na_spread():
    line = walkdown_line(_surface())
    assert line.endswith(" spread: n/a (fewer than two configurations to compare)")
    assert "band" not in line
    line = walkdown_line(_surface(S_naive=2.5, S_matched=2.1))  # the pooled spread is n/a: still no band
    assert "2.5x naive" in line and "band" not in line
    line = walkdown_line(_surface(S_naive=2.5, S_matched=2.1, S_honest=1.8))
    assert line.endswith("; 80% band 1.3x to 4.0x")


def _no_configuration(n_spread_draws, **over):
    """The tester's case: no configuration reaches the minimum, so the pooled spread is n/a. Some resamples
    can still give a spread (their band was 3.6x to 6.4x), so `n_boot_valid_S` may be 0 or above 0."""
    lo, hi = (math.nan, math.nan) if n_spread_draws == 0 else (3.6, 6.4)
    s = _surface(S_lo=lo, S_hi=hi, table=pd.DataFrame(), min_n=5, n_rows=5, n_boot=1000,
                 n_boot_valid_S=n_spread_draws, band_n_draws_min=None, band_n_invalid_draws_total=0,
                 band_support_note=_band_support_note(None, 1000, 0, n_spread_draws),
                 exclusions=[{"reason": "unknown acceptance", "n": 5}],
                 ratio_pair={"unavailable": "no workflow configuration met the minimum run count, so there "
                                            "is nothing to compare"})
    s.update(over)
    return s


def test_no_sentence_on_a_band_that_is_not_printed():
    for n_spread_draws in (0, 640):
        surface = _no_configuration(n_spread_draws)
        assert "spread band on the line above" in surface["band_support_note"]  # as the estimator wrote it
        text = render({}, {}, "", surface, walkdown_line(surface), [])
        assert "dollars spread: n/a" in text and "band" not in text, n_spread_draws
        assert "nothing to compare" not in text  # the empty table's own lines said why


def test_the_other_band_sentences_stay():
    note = _band_support_note(600, 1000, 3, 640)
    surface = _no_configuration(640, band_n_draws_min=600, band_n_invalid_draws_total=3, band_support_note=note)
    shown = shown_band_support_note(surface)
    assert "600 of 1000" in shown and "3 resampling draws were excluded" in shown and "line above" not in shown
    shown_band = _no_configuration(640, S_honest=4.1, band_support_note=note)  # the line prints its band
    assert shown_band_support_note(shown_band) == note and "; 80% band 3.6x to 6.4x" in walkdown_line(shown_band)


def test_an_empty_table_says_why_and_what_to_do():
    surface = {"table": pd.DataFrame(), "min_n": 5, "n_rows": 1234,
               "exclusions": [{"reason": "unknown acceptance", "n": 1200}, {"reason": "configuration below min_n", "n": 30},
                              {"reason": "no usable cost value", "n": 4}, {"reason": "no model label", "n": 0}]}
    lines = beat2_configurations(surface, "dollar spread: n/a", [])
    text = "\n".join(lines)
    assert "30 of 30 runs with known outcomes, usable cost and model labels" in lines[1]
    assert "minimum run count (--min-n 5)" in lines[1]
    assert "before comparison: 1,200 unknown acceptance, 4 no usable cost value" in text
    assert "no model label" not in text and "--min-n 3" in text



def test_json_and_home(tmp_path, capsys, monkeypatch):
    """`--home` puts the parse cache in the store's cache folder; `--json` is one object on stdout."""
    monkeypatch.delenv("LOOPMATH_CACHE_DIR", raising=False)
    monkeypatch.delenv("LOOPMATH_HOME", raising=False)
    logs = tmp_path / "logs" / "-repo"
    logs.mkdir(parents=True)
    shutil.copy(FIXTURES / "claude_code_sidechain.jsonl", logs / "s1.jsonl")
    store = tmp_path / "store"
    code = main(["analyze", "--logs", str(tmp_path / "logs"), "--all", "--home", str(store), "--json", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0 and [ln.split(":")[0] for ln in captured.err.splitlines()] == ["scan snapshot"]  # no timing
    out = json.loads(captured.out)
    assert out["schema"] == "loopmath.analyze/1" and out["min_n"] == 5
    for key in ("cost", "read", "coverage", "runs", "configurations", "spread", "excluded", "price_warnings"):
        assert key in out, key
    assert out["spread"]["band"] is None or out["spread"]["pooled"] is not None
    assert (store / "cache").is_dir() and not any(p.name != "cache" for p in store.iterdir())
    code = main(["analyze", "--logs", str(tmp_path / "logs"), "--all", "--home", str(store), "--json", "--quiet",
                 "--grading"])
    assert code == 0 and "grading" in json.loads(capsys.readouterr().out)


def test_json_has_no_band_for_an_na_spread():
    from loopmath.surface_presentation import walkdown

    for over, has_band in (({}, False), ({"S_naive": 2.5, "S_matched": 2.1}, False),
                           ({"S_naive": 2.5, "S_matched": 2.1, "S_honest": 1.8}, True)):
        surface = _surface(**over)
        out = analyze_payload(surface, diag={}, coverage={}, price_lines=[], grading=None, walkdown=walkdown(surface))
        assert (out["spread"]["band"] is not None) is has_band, over
        assert out["spread"]["pooled"] is None or not has_band or out["spread"]["pooled"] == 1.8


def test_empty_table_headline_names_the_dominant_overlap_exclusion():
    from loopmath.surface_estimation import cost_surface

    # Both substantial configurations share one cell, but need two for a
    # comparison. The small third configuration fails only the run minimum.
    rows = [{"run_id": f"{arm}-{i}", "arm": arm, "cell": "shared-task-mix",
             "usd": 0.1, "accepted": True, "tier": "reported", "workspace": "example/widget"}
            for arm, count in (("a", 12), ("b", 12), ("small", 3)) for i in range(count)]
    surface = cost_surface(pd.DataFrame(rows), min_n=5, min_overlap_cells=2, n_boot=0)
    surface["min_n"] = 5  # the analyze command attaches its CLI setting before rendering
    assert surface["table"].empty
    counts = {row["reason"]: row["n"] for row in surface["exclusions"]}
    assert counts["configuration below min_overlap_cells"] == 24
    assert counts["configuration below min_n"] == 3
    lines = beat2_configurations(surface, walkdown_line(surface), [])
    headline = lines[1].lower()
    assert "24" in headline and ("overlap" in headline or "task-mix" in headline), lines
    advice = "\n".join(lines).lower()
    assert "comparable" in advice or "shared" in advice or "same task" in advice
    assert "lower the minimum" not in advice


def test_cli_nondefault_minimum_agrees_in_terminal_json_and_html(tmp_path, monkeypatch, capsys):
    from loopmath.report import html, terminal
    from tests.test_ingest_ocp import _attempt, _doc

    attempts = [_attempt("analysis", suffix=f"{model}-{i}", model=model)
                for model, count in (("synthetic-a", 8), ("synthetic-b", 8), ("synthetic-c", 4))
                for i in range(count)]
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(_doc("analysis", attempts=attempts)))
    args = ["analyze", "--ocp", str(source), "--min-n", "9", "--boot", "0",
            "--quiet", "--home", str(tmp_path / "store")]
    assert main([*args, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["min_n"] == 9

    # analyze has no HTML flag. Render its actual CLI-computed surface through
    # the HTML renderer too, without substituting estimator or helper output.
    pages = []
    terminal_render = terminal.render

    def render_both(**kwargs):
        pages.append(html.render(**kwargs))
        return terminal_render(**kwargs)

    monkeypatch.setattr(terminal, "render", render_both)
    assert main(args) == 0
    text = capsys.readouterr().out
    explanation = payload["cheapest_vs_dearest"]["unavailable"]
    for output in (text, explanation, pages[0]):
        assert "20 of 20" in output
        assert "minimum run count (--min-n 9)" in output
        assert "minimum run count (--min-n 5)" not in output
