#!/usr/bin/env python3
"""Report disallowed prose classes without repeating matched text."""

from __future__ import annotations

import sys
from pathlib import Path

from loopmath.graph.ocp_support import _EM_DASH_RE


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: graph_html_prose_probe.py FILE_OR_DASH", file=sys.stderr)
        return 2
    text = sys.stdin.read() if sys.argv[1] == "-" else Path(sys.argv[1]).read_text()
    dash_count = len(_EM_DASH_RE.findall(text))
    # The vocabulary filter was lifted in 0.1.0 (Q3); only dashes are checked.
    clean = dash_count == 0
    print(
        f"{'PASS' if clean else 'FAIL'} prose scan: "
        f"dash_count={dash_count}"
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
