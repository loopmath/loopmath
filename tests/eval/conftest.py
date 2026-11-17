"""Shared pytest setup for the eval tests.

Recovery checks that need many fits are marked `slow` (PRD section 10). They are skipped
by default so the repository suite stays fast; set LOOPMATH_EVAL_SLOW=1 to run them.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: many-fit recovery check; runs only with LOOPMATH_EVAL_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LOOPMATH_EVAL_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow recovery check; set LOOPMATH_EVAL_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
