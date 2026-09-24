#!/usr/bin/env python3
"""Convert a herdr-dagr contract v1/v2/v3 run.json into an OCP v0.1 document.

The converter lives in the package as loopmath.ocp.contractv3; this example
is a thin CLI over it. `loopmath ocp migrate RUN.json` goes on to OCP v0.3.

    python3 spec/examples/ocp-from-contractv3.py IN_RUN_JSON [OUT_OCP_JSON]
"""

import sys
from pathlib import Path

# In a source checkout, the package next to this file wins over any installed loopmath.
_SRC = Path(__file__).resolve().parents[2] / "src"
if (_SRC / "loopmath" / "ocp" / "contractv3.py").is_file() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from loopmath.ocp.contractv3 import convert, main  # noqa: E402

__all__ = ["convert", "main"]

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
