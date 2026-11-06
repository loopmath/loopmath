"""Flows: an experiment on a fixed multi-step workflow.

A flow (an OCP 0.4 flow document) names the base configuration, the inputs, the acceptance rule, the objective,
the judges and the edits a planner may try. Its folder holds `flow.ocp.json` and `plan.json`. `load` reads a
folder or a flow document into a `FlowFolder`.
"""

from .folder import FlowFolder, load

__all__ = ["FlowFolder", "load"]
