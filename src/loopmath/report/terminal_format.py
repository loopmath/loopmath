"""Formatting primitives and configuration-display helpers for terminal reports."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal, Mapping, Sequence


def _fmt_money(value: float | int | None) -> str:
    """`$12.34` under $100, `$1,204` (no decimals) at or above $1,000."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    if abs(v) >= 1000:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def _fmt_pct(value: float | int | None) -> str:
    """`84%` from a 0..1 fraction. None/NaN render as `n/a`."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v * 100:.0f}%"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_plain_number(value: Any) -> str:
    """Comma-grouped number, no unit attached.

    Used only for a `cost_col` this module does not recognise: it must never
    guess a unit (no `$`, no "tokens" word), so it prints the number and lets
    the column header carry the raw column name instead.
    """
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"


# ---------------------------------------------------------------------------
# Width-aware table formatting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableColumn:
    """One column in a terminal table.

    ``priority`` controls narrow-terminal degradation: lower values are
    removed first, with a rightmost column removed first when priorities are
    equal. At least one column is always retained. ``align="auto"``
    right-aligns a column when all of its non-empty values are numeric.

    Tuples of ``(key, header, align)`` or
    ``(key, header, align, priority)`` are accepted too.
    """

    key: str
    header: str
    align: Literal["auto", "l", "r"] = "auto"
    priority: int = 0


ColumnSpec = TableColumn | tuple[str, str, str] | tuple[str, str, str, int]


def _display_width(value: str) -> int:
    """Return terminal-cell width without adding a wcwidth dependency."""
    width = 0
    for char in value:
        if unicodedata.combining(char) or char in {"\ufe0e", "\ufe0f", "\u200d"}:
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _truncate_display(value: str, width: int) -> str:
    """Truncate to terminal cells, preserving combining marks."""
    if _display_width(value) <= width:
        return value
    if width <= 0:
        return ""
    if width == 1:
        return "…"

    available = width - 1
    result: list[str] = []
    used = 0
    for char in value:
        char_width = _display_width(char)
        if char_width and used + char_width > available:
            break
        result.append(char)
        used += char_width
    return "".join(result).rstrip() + "…"


def _column(spec: ColumnSpec) -> TableColumn:
    if isinstance(spec, TableColumn):
        column = spec
    elif len(spec) == 3:
        column = TableColumn(*spec)
    elif len(spec) == 4:
        column = TableColumn(*spec)
    else:
        raise ValueError("table columns must have 3 or 4 entries")
    if column.align not in {"auto", "l", "r"}:
        raise ValueError(f"unknown table alignment: {column.align!r}")
    return column


def _cell_text(value: Any) -> str:
    """Make one physical terminal line from an arbitrary cell value."""
    return " ".join(str(value).replace("\t", " ").splitlines())


def render_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[ColumnSpec],
    width: int | None = None,
) -> list[str]:
    """Render an auto-width terminal table that never wraps at ``width``.

    Columns use their natural display-cell width. If the result is wider than
    ``width``, the lowest-priority column is dropped until it fits. Equal
    priorities drop from right to left. If even the final column is too wide,
    its cells are truncated with an ellipsis. A ``None`` width preserves the
    full natural-width output.
    """
    if not columns:
        return []
    if width is not None and width < 1:
        raise ValueError("table width must be at least 1")

    active = [_column(spec) for spec in columns]
    text_rows = [
        {column.key: _cell_text(row.get(column.key, "")) for column in active}
        for row in rows
    ]

    def _align(column: TableColumn) -> str:
        if column.align != "auto":
            return column.align
        values = [row.get(column.key) for row in rows]
        nonempty = [
            value
            for value in values
            if value is not None and not (isinstance(value, str) and value == "")
        ]
        return "r" if nonempty and all(isinstance(value, Number) for value in nonempty) else "l"

    def _widths(selected: Sequence[TableColumn]) -> list[int]:
        return [
            max(
                [_display_width(column.header)]
                + [_display_width(row[column.key]) for row in text_rows]
            )
            for column in selected
        ]

    col_widths = _widths(active)
    while width is not None and len(active) > 1 and sum(col_widths) + 2 * (len(active) - 1) > width:
        drop = min(range(len(active)), key=lambda i: (active[i].priority, -i))
        del active[drop]
        col_widths = _widths(active)

    if width is not None and len(active) == 1:
        col_widths[0] = min(col_widths[0], width)

    def _pad(value: str, cell_width: int, align: str) -> str:
        value = _truncate_display(value, cell_width)
        padding = " " * (cell_width - _display_width(value))
        return padding + value if align == "r" else value + padding

    lines: list[str] = []
    header_cells = [
        _pad(column.header, cell_width, _align(column))
        for column, cell_width in zip(active, col_widths)
    ]
    lines.append("  ".join(header_cells).rstrip())

    for row in text_rows:
        cells = [
            _pad(row[column.key], cell_width, _align(column))
            for column, cell_width in zip(active, col_widths)
        ]
        lines.append("  ".join(cells).rstrip())

    return lines
def _cost_col_presentation(cost_col: str) -> tuple[str, str, Any]:
    """Subtitle noun phrase, column header, and value formatter for a cost basis.

    SPEC section 5: dollars is the primary basis (`"usd"`), output tokens the
    secondary one (`"out_tokens"`); a report in token mode must never print a
    `$` in front of a token count. Any other `cost_col` is uncharted: name it
    plainly with its own raw column name instead of guessing which unit it
    is in.
    """
    if cost_col == "usd":
        return "dollars per accepted run", "$/accepted", _fmt_money
    if cost_col == "out_tokens":
        return "output tokens per accepted run", "tokens/accepted", _fmt_int
    return f"{cost_col} per accepted run", f"{cost_col}/accepted", _fmt_plain_number


def _ci_presentation(ci: Any) -> tuple[str, str]:
    """Confidence-band label and column header for whatever `ci` the surface used.

    `ci` should be the fraction (e.g. `0.80`) `cost_surface` was actually run
    with. A missing, non-numeric, NaN, or out-of-[0,1]-range value means the
    band width was never actually recorded against this table, so this says
    that plainly instead of asserting a width (like the old hardcoded 80%)
    that may not be the one the bands underneath were built with.
    """
    ci_val: float | None
    try:
        ci_val = float(ci)
    except (TypeError, ValueError):
        ci_val = None
    if ci_val is not None and (ci_val != ci_val or not (0 <= ci_val <= 1)):
        ci_val = None
    if ci_val is None:
        fallback = "confidence band (width not recorded)"
        return fallback, fallback
    return f"{ci_val:.0%} confidence band", f"{ci_val:.0%} band"
