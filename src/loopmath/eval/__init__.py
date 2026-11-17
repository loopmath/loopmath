"""`loopmath eval` (experimental): rubric scores from judges and reviewers, read as noisy measurements.

Each command is one stage: named files in, one file out, a versioned schema between stages. The package
imports nothing from `store`, `belief`, `flows`, `compare`, `recommend` or `gate`; only `adapters/ocp.py`
reads OCP documents, and only `engine_pymc.py` imports PyMC. `tests/eval/test_eval_import_rule.py` checks it.
"""
