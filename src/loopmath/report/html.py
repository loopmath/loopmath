"""Self-contained HTML renderer for an analyzed loopmath cost surface.

Run ``python -m loopmath.report.html --demo`` to write ``loopmath-report.html`` in
the current directory.  The result contains its stylesheet and all report
content inline, so it can be opened without a server or network connection.
"""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .terminal import PRIVACY_LINE, beat3_exclusions


_STYLE = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  color: #172026; background: #f4f1ea; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1rem; }
main { max-width: 1050px; margin: auto; }
header { border-bottom: 1px solid #c9c4b9; padding-bottom: 1.2rem; }
h1 { margin: 0 0 .3rem; font-size: clamp(2rem, 5vw, 3.6rem); letter-spacing: -.045em; }
h2 { margin: 0 0 1rem; font-size: 1.25rem; }
p { line-height: 1.55; }
.eyebrow { color: #27615b; font-size: .78rem; font-weight: 750; letter-spacing: .12em;
  text-transform: uppercase; }
.coverage { margin-top: 1rem; padding: .85rem 1rem; border-left: 4px solid #d96c47;
  background: #fffdf8; font-weight: 650; }
.panel { margin-top: 1rem; padding: 1.3rem; border: 1px solid #d5d0c5;
  border-radius: 12px; background: #fffdf8; box-shadow: 0 5px 18px #1720260b; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: .8rem; margin-top: 1rem; }
.summary { padding: .85rem; border-radius: 8px; background: #edf2ee; }
.summary span { display: block; color: #53615e; font-size: .78rem; }
.summary strong { display: block; margin-top: .18rem; font-size: 1.25rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: .8rem .65rem; border-bottom: 1px solid #ded9ce; text-align: right;
  vertical-align: middle; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: #53615e; font-size: .75rem; letter-spacing: .06em; text-transform: uppercase; }
.band-cell { min-width: 190px; }
.band-track { position: relative; width: 100%; height: 10px; border-radius: 5px;
  background: #e4e0d6; }
.band-range { position: absolute; height: 10px; border-radius: 5px; background: #72a69d; }
.band-point { position: absolute; top: -3px; width: 3px; height: 16px; background: #172026; }
.band-label { margin-top: .3rem; color: #53615e; font-size: .75rem; }
.ratios { display: flex; flex-wrap: wrap; gap: .8rem; margin-top: 1rem; }
.ratio { min-width: 150px; padding: .8rem 1rem; border-radius: 8px; background: #172026;
  color: white; }
.ratio span { display: block; color: #b9cbc7; font-size: .75rem; }
.ratio strong { font-size: 1.35rem; }
.walkdown { font-size: 1.05rem; font-weight: 650; }
.note { color: #53615e; font-size: .9rem; }
ul { margin: 0; padding-left: 1.2rem; }
li { margin: .45rem 0; line-height: 1.45; }
footer { margin-top: 1rem; padding: 1.1rem; text-align: center; color: #27615b;
  font-weight: 750; }
@media (max-width: 650px) { body { padding: 1rem .6rem; } .panel { padding: 1rem .75rem; }
  th, td { padding: .65rem .45rem; } }
"""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _money(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"${number:,.0f}" if abs(number) >= 1000 else f"${number:,.2f}"


def _integer(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:,.0f}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.0%}"


def _rows(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "iterrows"):
        return [dict(row) for _, row in table.iterrows()]
    if isinstance(table, dict):
        return [table]
    try:
        return [dict(row) for row in table]
    except (TypeError, ValueError):
        return []


def _band_positions(rows: Iterable[dict[str, Any]]) -> tuple[float, float]:
    values = [
        value
        for row in rows
        for value in (
            _number(row.get("lo")),
            _number(row.get("hi")),
            _number(row.get("cost_per_accepted")),
        )
        if value is not None
    ]
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    return (low, high) if high > low else (low, low + 1.0)


def _configuration_table(surface: dict[str, Any]) -> str:
    def _sort_value(row: dict[str, Any]) -> float:
        value = _number(row.get("cost_per_accepted"))
        return value if value is not None else float("inf")

    rows = sorted(
        _rows(surface.get("table")),
        key=_sort_value,
    )
    if not rows:
        return '<p class="note">No configurations met the minimum run count.</p>'

    scale_low, scale_high = _band_positions(rows)
    span = scale_high - scale_low
    body = []
    for row in rows:
        point = _number(row.get("cost_per_accepted"))
        lo = _number(row.get("lo"))
        hi = _number(row.get("hi"))
        if lo is None or hi is None:
            band = '<span class="note">n/a</span>'
        else:
            left = max(0.0, min(100.0, (lo - scale_low) / span * 100))
            right = max(left, min(100.0, (hi - scale_low) / span * 100))
            width = max(1.5, right - left)
            point_pos = (
                left
                if point is None
                else max(0.0, min(100.0, (point - scale_low) / span * 100))
            )
            band = (
                f'<div class="band-track" role="img" aria-label="80% band {_money(lo)} to {_money(hi)}">'
                f'<span class="band-range" style="left:{left:.2f}%;width:{width:.2f}%"></span>'
                f'<span class="band-point" style="left:{point_pos:.2f}%"></span></div>'
                f'<div class="band-label">{_money(lo)} to {_money(hi)}</div>'
            )
        name = escape(str(row.get("arm", "?")))
        tiers = escape(str(row.get("tiers") or "n/a"))
        body.append(
            "<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{_integer(row.get('n'))}</td>"
            f"<td>{_percent(row.get('acc_rate'))}</td>"
            f"<td>{tiers}</td>"
            f"<td><strong>{_money(point)}</strong></td>"
            f'<td class="band-cell">{band}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Configuration</th><th>Runs</th><th>Accepted</th><th>Tiers</th>"
        "<th>$/accepted</th><th>80% band</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _ratio_html(surface: dict[str, Any]) -> str:
    pair = surface.get("ratio_pair") or {}
    if pair.get("unavailable"):
        return f'<p class="note">{escape(str(pair["unavailable"]))}</p>'
    if not pair:
        return '<p class="note">The lowest and highest cost configurations were not compared.</p>'
    token = _number(pair.get("token_ratio"))
    dollar = _number(pair.get("dollar_ratio"))
    amplifier = _number(pair.get("amplifier"))
    cards = [
        '<div class="ratio"><span>Token ratio</span><strong>'
        + (f"{token:.1f}x" if token is not None else "n/a")
        + "</strong></div>",
        '<div class="ratio"><span>Dollar ratio</span><strong>'
        + (f"{dollar:.1f}x" if dollar is not None else "n/a")
        + "</strong></div>",
    ]
    if amplifier is not None:
        cards.append(
            f'<div class="ratio"><span>Amplifier</span>'
            f"<strong>{amplifier:.1f}x</strong></div>"
        )
    note = pair.get("basis_note")
    suffix = f'<p class="note">{escape(str(note))}</p>' if note else ""
    return '<div class="ratios">' + "".join(cards) + "</div>" + suffix


def render(
    *,
    ingest_diag: dict | None = None,
    coverage: dict | None = None,
    coverage_line: str = "",
    surface: dict | None = None,
    walkdown_line: str = "",
    price_warnings: list[str] | None = None,
    title: str = "loopmath cost report",
) -> str:
    """Return one complete HTML document for already-analyzed data."""
    ingest_diag, coverage, surface = ingest_diag or {}, coverage or {}, surface or {}
    records = ingest_diag.get("records") or {}
    skipped = ingest_diag.get("skipped") or {}
    tiers = coverage.get("tiers") or {}
    structural_pct = _number(coverage.get("pct_structural"))
    summaries = [
        ("Runs read", records.get("total")),
        ("Files skipped", skipped.get("total")),
        ("Verified", tiers.get("verified")),
        (
            "Structurally graded",
            f"{structural_pct:.1f}%" if structural_pct is not None else "n/a",
        ),
    ]
    summary_html = "".join(
        f'<div class="summary"><span>{escape(label)}</span>'
        f'<strong>{escape(str(value if value is not None else "n/a"))}</strong></div>'
        for label, value in summaries
    )
    exclusion_lines = beat3_exclusions(surface, coverage, ingest_diag)[1:]
    exclusions = "".join(f"<li>{escape(line.strip())}</li>" for line in exclusion_lines)
    notices = [
        surface.get(key)
        for key in ("tier_note", "overlap_caveat", "band_note", "band_support_note")
    ]
    notices.extend(price_warnings or [])
    notes_html = "".join(f'<p class="note">{escape(str(note))}</p>' for note in notices if note)
    coverage_text = coverage_line or "No grading coverage available."
    walkdown_text = walkdown_line or "No cost walkdown available."

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body><main>"
        f'<header><div class="eyebrow">Local workflow analysis</div><h1>{escape(title)}</h1>'
        '<p>Cost per accepted run, with uncertainty and evidence coverage kept visible.</p></header>'
        f'<section class="panel"><h2>What I read</h2><div class="summary-grid">{summary_html}</div>'
        f'<p class="coverage">{escape(coverage_text)}</p></section>'
        f'<section class="panel"><h2>Configurations</h2>{_configuration_table(surface)}'
        f'{_ratio_html(surface)}{notes_html}</section>'
        f'<section class="panel"><h2>Cost walkdown</h2><p class="walkdown">{escape(walkdown_text)}</p></section>'
        f'<section class="panel"><h2>What I left out</h2><ul>{exclusions}</ul></section>'
        f'<footer>{escape(PRIVACY_LINE)}</footer></main></body></html>\n'
    )


def demo_data() -> dict[str, Any]:
    """Small deterministic input used by the documented offline demo."""
    return {
        "ingest_diag": {
            "records": {"claude-code": 82, "codex": 54, "total": 136},
            "skipped": {"total": 7},
            "skip_reasons": {"unreadable or empty": 7},
        },
        "coverage": {
            "pct_structural": 91.2,
            "tiers": {
                "verified": 91,
                "reported": 18,
                "heuristic": 15,
                "asserted": 5,
                "censored": 7,
            },
        },
        "coverage_line": "graded 124 of 136 runs (91.2% structural); tiers: verified 91 / reported 18 / heuristic 15 / asserted 5 / censored 7",
        "surface": {
            "table": [
                {
                    "arm": "sol-medium",
                    "n": 48,
                    "acc_rate": 0.81,
                    "tiers": "31v/9r/8h",
                    "cost_per_accepted": 4.82,
                    "lo": 4.10,
                    "hi": 5.73,
                },
                {
                    "arm": "opus-xhigh",
                    "n": 39,
                    "acc_rate": 0.87,
                    "tiers": "33v/4r/2h",
                    "cost_per_accepted": 16.40,
                    "lo": 13.20,
                    "hi": 20.10,
                },
                {
                    "arm": "terra-high",
                    "n": 37,
                    "acc_rate": 0.76,
                    "tiers": "27v/5r/5h",
                    "cost_per_accepted": 8.65,
                    "lo": 7.30,
                    "hi": 10.90,
                },
            ],
            "ratio_pair": {"token_ratio": 3.1, "dollar_ratio": 3.4, "amplifier": 1.1},
            "exclusions": [
                {
                    "reason": "unknown acceptance",
                    "n": 12,
                    "detail": "runs with unknown acceptance",
                }
            ],
        },
        "walkdown_line": "naive spread 4.1x; honest, cell-adjusted spread 3.4x (80% band 2.8x to 4.2x)",
        "price_warnings": ["Price table as of 2026-08-31."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a self-contained loopmath HTML report")
    parser.add_argument("--demo", action="store_true", help="render deterministic example data")
    parser.add_argument(
        "--output",
        default="loopmath-report.html",
        help="output file (default: loopmath-report.html)",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("--demo is required when running this module directly")
    output = Path(args.output)
    output.write_text(render(**demo_data()), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
