"""The terminal report's read-summary beat."""

from __future__ import annotations

from .terminal_format import _fmt_int, render_table


_HARNESS_ORDER = ("claude-code", "codex")


def _counted_noun(count: int | float, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    number = _fmt_int(count) if float(count).is_integer() else f"{float(count):g}"
    return f"{number} {word}"


def beat1_what_i_read(ingest_diag: dict, coverage: dict, coverage_line: str) -> list[str]:
    """How many session files were seen, parsed, and skipped, then coverage."""
    lines = ["What I read"]

    diag = ingest_diag or {}
    files_seen = diag.get("files_seen") or {}
    records = diag.get("records") or {}
    skipped = diag.get("skipped") or {}
    grouped = diag.get("grouped_continuation_files") or {}
    cache_hits = diag.get("cache_hits")
    ocp_documents = diag.get("ocp_documents", 0)
    ocp_attempts = diag.get("ocp_attempts", 0)

    if not files_seen and not records and not skipped and not ocp_documents:
        lines.append("  no session files seen")
    else:
        # Prose lines ("N session files, N parsed, N skipped") per harness,
        # per SPEC's example tone; the label column is padded so the numbers
        # line up without building a boxed table for four loose columns.
        rows_for_prose = []
        for harness in _HARNESS_ORDER:
            if harness not in files_seen and harness not in records and harness not in skipped:
                continue
            grouped_count = grouped.get(harness, 0)
            text = (
                f"{_counted_noun(files_seen.get(harness, 0), 'session file')}, "
                f"{_fmt_int(records.get(harness, 0))} parsed, "
                f"{_fmt_int(skipped.get(harness, 0))} skipped"
            )
            if grouped_count:
                earlier = "an earlier file" if grouped_count == 1 else "earlier files"
                text += (
                    f", {_counted_noun(grouped_count, 'continuation file')} "
                    f"combined with {earlier}"
                )
            rows_for_prose.append(
                {
                    "label": harness,
                    "text": text,
                }
            )
        native_sources = bool(files_seen or records or skipped)
        if native_sources:
            grouped_total = grouped.get("total", 0)
            total_text = (
                f"{_counted_noun(files_seen.get('total', 0), 'session file')}, "
                f"{_fmt_int(records.get('total', 0))} parsed, "
                f"{_fmt_int(skipped.get('total', 0))} skipped"
            )
            if grouped_total:
                earlier = "an earlier file" if grouped_total == 1 else "earlier files"
                total_text += (
                    f", {_counted_noun(grouped_total, 'continuation file')} "
                    f"combined with {earlier}"
                )
            if cache_hits:
                total_text += f" ({_fmt_int(cache_hits)} from cache)"
            rows_for_prose.append({"label": "total", "text": total_text})
        if ocp_documents:
            rows_for_prose.append(
                {
                    "label": "ocp",
                    "text": (
                        f"{_fmt_int(ocp_documents)} OCP document(s), "
                        f"{_counted_noun(ocp_attempts, 'attempt')} imported"
                    ),
                }
            )

        # The time window is part of what was read, so it belongs here rather
        # than in a footnote. Without it, a reader sees the parsed count and
        # has no way to know a default window kept most of the estate out.
        if native_sources:
            since_days = diag.get("since_days")
            if since_days is None:
                window_text = "every log file, no time limit"
            else:
                n_days = int(since_days) if float(since_days) == int(since_days) else since_days
                window_text = f"log files modified in the last {_counted_noun(n_days, 'day')}"
                outside = diag.get("files_outside_window")
                if outside:
                    verb = "was" if outside == 1 else "were"
                    window_text += (
                        f"; {_counted_noun(outside, 'older file')} {verb} not read "
                        "(pass --all to read every one)"
                    )
            rows_for_prose.append({"label": "window", "text": window_text})

        # FIX 2 (reviewer round, blocker): `--limit` truncates the discovered
        # file list before anything is parsed; without this line the counts
        # above describe that truncated list as if it were the whole corpus,
        # with no sign anything was cut. Same style as the `window` line
        # above: only appears when something was actually removed.
        limit = diag.get("limit")
        omitted_by_limit = (diag.get("files_omitted_by_limit") or {}).get("total")
        if limit is not None and omitted_by_limit:
            verb = "was" if omitted_by_limit == 1 else "were"
            omitted_pronoun = "it" if omitted_by_limit == 1 else "them"
            rows_for_prose.append(
                {
                    "label": "limit",
                    "text": (
                        f"--limit {_fmt_int(limit)} per harness; "
                        f"{_counted_noun(omitted_by_limit, 'more file')} {verb} found but not "
                        "read because of it (raise --limit, or drop it, to read "
                        f"{omitted_pronoun})"
                    ),
                }
            )

        # Coordination note (reviewer round): another module may count
        # zero-token synthetic sessions under `zero_token_synthetic`. Printed
        # only when the key is present and greater than zero, so an ingest
        # diagnostic that has not wired this up yet renders exactly as
        # before with no missing-key error and no fabricated line.
        zero_token_synthetic = diag.get("zero_token_synthetic")
        if zero_token_synthetic:
            rows_for_prose.append(
                {
                    "label": "zero-token",
                    "text": (
                        f"{_counted_noun(zero_token_synthetic, 'parsed session')} carried an "
                        "all-zero token count"
                    ),
                }
            )

        label_w = max(len(r["label"]) for r in rows_for_prose)
        for r in rows_for_prose:
            lines.append(f"  {r['label'].ljust(label_w)}   {r['text']}")

    lines.append("")
    lines.append(f"  {coverage_line}" if coverage_line else "  no grading coverage available")

    # The coverage line above already names every tier inline (grade.py's own
    # wording); this second, vertically aligned breakdown exists so the tier
    # counts stay visible and easy to scan even if that wording ever changes.
    # SPEC section 0: evidence tiers print in output, never hidden.
    tiers = (coverage or {}).get("tiers") or {}
    # `ungraded` is last because it is not an evidence tier, it is the count
    # of parsed runs no tier applied to. Leaving it out of this table was
    # the reason the printed tiers did not sum to the runs read: a reader
    # could add the column up, get a smaller number, and have nothing on
    # screen to explain the difference.
    tier_order = ["verified", "reported", "heuristic", "asserted", "censored", "ungraded"]
    tier_rows = [
        {"tier": t, "count": _fmt_int(tiers.get(t, 0))} for t in tier_order if t in tiers
    ]
    if tier_rows:
        lines.append("")
        tier_lines = render_table(
            tier_rows, [("tier", "tier", "l"), ("count", "runs", "r")]
        )
        lines.extend(f"  {line}" for line in tier_lines)

    return lines
