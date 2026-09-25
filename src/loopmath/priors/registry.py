"""Prior sources: names, default inputs and the hand labels for the sweep tasks (lane 11).

Each bundled run carries its source as `run.task.source.kind`, which is also the
source node `fit --without SOURCE` drops (spec 04, section 5). `benchmark` is
reserved for published results used as prior factors.

Not named `sources.py`: a submodule of that name would replace the read API's
`loopmath.priors.sources()` once imported.
"""

from __future__ import annotations

import os
from pathlib import Path

SWEEP = "sweep"
E0 = "e0"
RQ1 = "rq1"
REPO_HISTORY = "repo_history"
BENCHMARK = "benchmark"
BUNDLE_SOURCES = (SWEEP, E0, RQ1, REPO_HISTORY)

# Public repo names that stay readable after the share reduction (spec 03,
# section 7 hashes every other repo). Both are our own benchmark task sets.
PUBLIC_REPOS = ("loopmath-sweep", "ale-bench")

# Inputs on the build machine, read only; nothing at run time needs them (the
# bundle ships built). Each is an explicit path, the `prior build` flag or else
# the environment variable, with no default folder. The sweep and E0
# variables are the ones `research fit` and `analyze-e0` read.
INPUT_FLAGS = {SWEEP: "--sweep-dir", E0: "--e0-corpus", RQ1: "--rq1-dir"}
ENV_INPUTS = {SWEEP: "LOOPMATH_SWEEP_DIR", E0: "LOOPMATH_E0_CORPUS", RQ1: "LOOPMATH_PRIOR_RQ1"}


class MissingInput(LookupError):
    """No path was given for a prior source."""


def input_path(name: str, override: str | Path | None = None) -> Path:
    """The input folder for a source: `override`, else its environment variable, else MissingInput."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get(ENV_INPUTS[name])
    if env:
        return Path(env).expanduser()
    raise MissingInput(f"no input folder for prior source {name}: pass {INPUT_FLAGS[name]} PATH "
                       f"or set {ENV_INPUTS[name]}")


def _task(type_: str, size: str, changed: int, *, touches: str = "one", design: str = "no") -> dict:
    return {"type": type_, "changed_lines": changed,
            "features": {"size": size, "lang": "python", "has_tests": "yes", "spec_clarity": "clear",
                         "needs_design": design, "touches": touches}}


# Labelled by lane 11 from each task's task.md; `changed_lines` is the measured
# starter-to-reference diff (`diff -r starter reference`, 2026-09-23).
SWEEP_TASKS: dict[str, dict] = {
    "t1-cli-tool": _task("feature", "m", 117),
    "t2-algorithm": _task("feature", "s", 72),
    "t3-bugfix": _task("bug_fix", "s", 43),
    "t4-refactor": _task("refactor", "m", 161, touches="few", design="yes"),
    "t5-parser-api": _task("feature", "m", 205),
    "t6-hard": _task("feature", "l", 609, design="yes"),
    "t7-dataflow": _task("feature", "l", 797, design="yes"),
    "t7-regex": _task("feature", "l", 663, design="yes"),
    "t7-rope": _task("feature", "l", 550, design="yes"),
    "t7-sched": _task("feature", "l", 568, design="yes"),
    "t7-sql": _task("feature", "l", 1146, design="yes"),
}
