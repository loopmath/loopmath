"""One compile check for the tests that run pytensor's C backend (the PyMC checks in tests/test_fit.py and
tests/belief/test_belief_fit.py): they skip only when pytensor cannot compile C code on this machine, and any other
failure is the test's own. The session's compile folder is the one tests/conftest.py sets."""

from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def pytensor_compile_error() -> str | None:
    """None when pytensor compiles C code here, in the CVM backend `fit_bayes.sample` and `check_pymc` pin; else the
    first error line of its compile failure (for example a clang that rejects a flag pytensor passes). Without
    pytensor, None: the test meets that itself."""
    try:
        import pytensor
        import pytensor.tensor as pt
        from pytensor.link.c.exceptions import CompileError
    except ImportError:
        return None
    try:
        with pytensor.config.change_flags(mode="CVM"):
            x = pt.dscalar("x")
            pytensor.function([x], x + 1.0)(1.0)
    except CompileError as exc:
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        first = [ln for ln in lines if ln.startswith("ld:")] + [ln for ln in lines if "error:" in ln] + lines
        return first[0] if first else "no message"
    return None


def skip_unless_pytensor_compiles() -> None:
    """Skip when pytensor cannot compile on this machine; any other failure is the test's own."""
    reason = pytensor_compile_error()
    if reason:
        pytest.skip(f"pytensor cannot compile C code on this machine: {reason}")
