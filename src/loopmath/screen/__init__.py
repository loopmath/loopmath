"""`loopmath screen` (experimental): a fold-over screen of atomic harness edits.

Up to about 16 candidate edits (CLAUDE.md lines, skills, hooks, scripts) go through one design. Each pair runs one
task twice: arm A with the edits its row turns on, arm B with the others. Task difficulty cancels in the pair's
difference, so every session informs every edit. A Bayesian linear model on the differences ranks the edits on
success and cost with ranges, and predicts every subset, so the screen names the predicted-best harness, often
one nobody ran. `design` builds the rows, `model` fits them, `edits` builds the arms' trees, `record` keeps the
screen in the store and `commands` is the CLI.
"""

SCREEN_SCHEMA = "loopmath.screen/1"
SCREEN_EXT = "dev.loopmath.screen"  # run.ext key: {"screen": scr_..., "arm": "A"|"B", "pair": "p01", "on": [...]}
ARMS = ("A", "B")
