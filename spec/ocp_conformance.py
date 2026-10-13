#!/usr/bin/env python3
"""Shim: the OCP conformance checker now lives in loopmath.ocp.conformance (spec 01 section 3).

Kept so `python3 spec/ocp_conformance.py FILE ...` and code that loads this
file by path keep working. Every name of the package module is re-exported.
"""

import sys
from pathlib import Path

# In a source checkout, the package next to this file wins over any installed
# loopmath (an editable install of another checkout would otherwise be used).
_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "loopmath" / "ocp" / "conformance.py").is_file() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from loopmath.ocp import conformance as _checker  # noqa: E402

globals().update({k: v for k, v in vars(_checker).items() if not k.startswith("__")})
__doc__ = _checker.__doc__

if __name__ == "__main__":
    sys.exit(_checker.main(sys.argv[1:]))
