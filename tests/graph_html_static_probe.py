#!/usr/bin/env python3
"""Static acceptance checks for a rendered graph HTML file.

`network_findings` is the one network scan: the probe and the viewer tests both
call it, so the page is held to a single definition of reaching the network.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


# Elements whose whole job is to load something from wherever their source
# attribute points. A clean page has none of them; a `script` counts only when
# it names a `src`, since the viewer's own code is inline.
LIVE_TAGS = ("link", "img", "iframe", "object", "embed", "audio", "video", "source")
_ATTR_URL = re.compile(
    r"(?i)https?://[^\s\"'<>]+|(?<![:\w/])//[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s\"'<>]*)?"
)


def _cut(value: str | None, limit: int = 60) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


class _NetworkScan(HTMLParser):
    """Collect the live markup through which a page could reach the network.

    Only markup counts: a `script` that names a `src`, an element that exists to
    load a resource, an inline event handler, an absolute `http` or `https` URL
    (or its protocol-relative form) inside a tag attribute. Text is not markup,
    so an attempt whose transcript says `fetch(`, `XMLHttpRequest` or
    `WebSocket` is not a finding. Script and style bodies are character data to
    this parser, which is what keeps the embedded payload out of the scan.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.findings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and "src" in values:
            self.findings.append(f"<script> with src {_cut(values['src'])}")
        elif tag in LIVE_TAGS:
            self.findings.append(f"<{tag}> element")
        for name, value in attrs:
            if name.startswith("on"):
                self.findings.append(f"inline event handler {name} on <{tag}>")
            found = _ATTR_URL.search(value or "")
            if found:
                self.findings.append(
                    f"absolute URL in <{tag}> {name}: {_cut(found.group(0))}"
                )


def network_findings(page: str) -> list[str]:
    """Every live element or attribute in `page` that could reach the network."""
    scan = _NetworkScan()
    scan.feed(page)
    scan.close()
    return scan.findings


EXPECTED_COLUMNS = (
    "attempt",
    "role",
    "model",
    "tier",
    "cost",
    "tokens",
    "duration",
    "status",
    "artifacts written",
    "artifacts read",
)
EXPECTED_VIEWS = ("swim", "force", "cost")
LIMIT = 500 * 1024


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: graph_html_static_probe.py FILE", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    raw = path.read_bytes()
    page = raw.decode("utf-8")
    failures = 0

    def check(name: str, condition: bool, detail: str) -> None:
        nonlocal failures
        print(f"{'PASS' if condition else 'FAIL'} {name}: {detail}")
        if not condition:
            failures += 1

    findings = network_findings(page)
    check(
        "offline content",
        not findings,
        "no live element or attribute" if not findings else "; ".join(findings),
    )

    block = re.search(r"const COLS = \[(.*?)\n  \];", page, re.S)
    labels = tuple(re.findall(r"label:\s*'([^']+)'", block.group(1))) if block else ()
    missing = [label for label in EXPECTED_COLUMNS if label not in labels]
    extra = [label for label in labels if label not in EXPECTED_COLUMNS]
    check(
        "table columns",
        labels == EXPECTED_COLUMNS,
        f"missing={missing} extra={extra} order={list(labels)}",
    )

    details = re.findall(
        r'<details class="viewsec" data-view="([^"]+)"([^>]*)>', page
    )
    views = tuple(item[0] for item in details)
    check("view order", views == EXPECTED_VIEWS, str(list(views)))
    folded = [name for name, attrs in details if "open" not in attrs.split()]
    missing_views = [name for name in EXPECTED_VIEWS if name not in views]
    fold_detail = (
        "all three independently foldable"
        if not folded and not missing_views
        else f"missing_details={missing_views} not_initially_open={folded}"
    )
    check(
        "fold controls",
        len(details) == 3 and not folded,
        fold_detail,
    )

    check("size", len(raw) < LIMIT, f"{len(raw)} bytes, limit {LIMIT - 1}")
    print(f"SUMMARY failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
