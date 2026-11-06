"""The generic run log `loopmath.harness-runs/0.1` and its importer (spec/harness-runs.md, docs/adapters.md).

One JSON line per rollout: the task, the harness that ran it, the outcome and the cost. `to_ocp` turns a row
into an OCP v0.3 run that `loopmath run import` stores, or, for a row naming a flow and given that flow, an OCP
0.4 run of the flow's arm; `to_ocp_rows` converts several rows at once. The harness2 reader
(`loopmath.adapters.harness2`) writes these rows, and so can any tool or a hand A/B. Nothing in loopmath's model
path imports this package.
"""

from __future__ import annotations

from .costs import price_tokens, rollout_cost
from .ocp import NS, PAIR_KEY, arms, flow_row_findings, run_id_for, to_ocp, to_ocp_rows
from .rows import MAX_ITEMS, MAX_KEPT_RUN_ID, SCHEMA, Locked, append, check_row, locked, read, row, row_id
from .streams import rollout_usage

__all__ = ["Locked", "MAX_ITEMS", "MAX_KEPT_RUN_ID", "NS", "PAIR_KEY", "SCHEMA", "append", "arms", "check_row",
           "flow_row_findings", "locked", "price_tokens", "read", "rollout_cost", "rollout_usage", "row", "row_id",
           "run_id_for", "to_ocp", "to_ocp_rows"]
