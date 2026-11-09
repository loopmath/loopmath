"""`loopmath frontier` (experimental): find the cost and pass-rate frontier of a set of settings by ask and tell.

`init` declares the tasks, the settings and a budget; `next` gives one trial, a setting chosen by KG per dollar and a
task chosen by the task chooser; `record` takes the result back as an ordinary store run; `report` prints each
setting on the estimated frontier with P(on frontier), labelled a model probability not yet checked for calibration.

`model` is the study's task posterior and learned effort prior, `kg` the kg_dollar picker, `pareto` the frontier
masks, `cost` the cost model behind one protocol (`CostModel`), `strategy` the pick seam (`Strategy`, `TaskChooser`,
`STRATEGIES`), `state` the frontier file and `commands` the CLI. Ported from the DeepSWE frontier study.
"""

FRONTIER_SCHEMA = "loopmath.frontier/1"
FRONTIER_EXT = "dev.loopmath.frontier"  # run.ext key: {"frontier": frn_..., "trial": trl_..., "setting", "task"}
