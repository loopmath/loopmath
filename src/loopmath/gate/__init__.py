"""The adoption gate: before a loop adopts a change, is the gain real and worth its cost?

`decide(base_runs, cand_runs, policy)` answers adopt, reject (keep the base) or replicate N more
pairs, with the reasons and the cost, on top of `loopmath.compare.paired`. It reads OCP run
documents, `loopmath.harness-runs/0.1` rows or compare's observations, and never writes anything.
"""

from __future__ import annotations

from .core import SCHEMA, Decision, GatePolicy, decide, p_adopt

__all__ = ["SCHEMA", "Decision", "GatePolicy", "decide", "p_adopt"]
