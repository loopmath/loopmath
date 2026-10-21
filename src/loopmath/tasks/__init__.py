"""Task picking: score a candidate on fewer tasks of a suite (`loopmath tasks pick`, experimental).

The history of other arms on the suite's tasks gives each task's difficulty and slope (`fit`). The
candidate's own runs give its ability, and with the history its suite score: the mean over the suite of its
chance of acceptance per task, with a range (`posterior`). `select` picks the next tasks and says when to
stop. Nothing here reads or changes the belief engine; `recommend` and `posterior` never import it.
"""

from .fit import Bank, TaskFit, fit_bank
from .posterior import CandidatePosterior
from .select import Decision, decide, next_tasks

__all__ = ["Bank", "TaskFit", "fit_bank", "CandidatePosterior", "Decision", "decide", "next_tasks"]
