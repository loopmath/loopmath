"""Model labeler and scorer for the labeling eval (EXTRACTOR-SPEC.md section 8, D2).

The labeler reads `dataset.jsonl` (written by `loopmath.graph.dataset`), batches the node items
in file order, `BATCH_SIZE` per call, and builds one prompt per batch (`PROMPT_VERSION`,
`build_prompt`): the node's skeleton plus its small text spans, no gold, no attempt list,
no E2 record. The model answers with one JSON object; `parse_response` turns it into
prediction items, one per node the batch asked about, every one at evidence tier
`reported` with the model, effort and prompt version named. Values outside the label
vocabulary are counted and left unpredicted, never coerced; ids the batch did not ask
about are counted, never emitted; ids the answer left out are counted.

`score(items, predictions)` compares the predictions with the gold labels of the node
and edge items and returns every figure with the counts it was computed from: role
accuracy (and fine role accuracy) over every gold-labeled node, boundary recall (gold
boundaries are the starts of every contract-v3 attempt on a session after its first, in
the order of `started_at`; a predicted boundary list is scored by count and, separately,
by time within `BOUNDARY_TOLERANCE_S`, since the ledger clock is not aligned with the
session clock; a node with an attempt start that did not parse has unknown gold and is
counted, not scored), edge precision and recall twice (a predicted parent is an edge into
the node, scored against the gold parent label; and the dataset's candidate edge items,
scored on `gold_edge` against the predicted parents), send-back and approval detection
twice (node-level gold, and every contract-v3 attempt on the node). An abstention (no
prediction, or null) is a wrong answer for every score and is counted apart so the reader
sees how many there were. A figure whose denominator is zero is `None` with a reason,
never `0`. The scores file carries nothing but the inputs' paths and the figures, so two
runs over the same inputs write the same bytes.

Per-dollar and per-second figures come from loopmath's own pricing of the labeler's sessions
(`price_sessions`: the codex rollout or Claude Code session each call left under the log
roots, parsed by `loopmath.ingest` and priced by `loopmath.price`), never from an estimate; a
session that cannot be found or priced is reported as such. The 0.5 percent cost bound
(`COST_BOUND`) is measured against the priced cost of the sessions the dataset labels and
compared, with the unpriced ones counted.

No network call happens here (spec section 0, rule 3): the model call is made by
`tools/grid.sh` through `codex exec` or `claude -p`; this module writes prompts, parses
answers, scores and prices.

CLI (`python -m loopmath.graph.labeler <verb>`):
  batches --dataset F [--count]                 batch count (and ids per batch)
  prompt --dataset F --batch N --out P [--prompt-version v1|v2|v3|v3-strict|v4]
                                                write the prompt of batch N
  parse --dataset F --batch N --response P --model M --effort E --out P [--claude-json]
        [--prompt-version v1|v2|v3|v3-strict|v4]
                                                append the parsed predictions to P
  session (--events P | --claude-json P)        the session id a call left behind
  find-session --newer-than P --cwd D --harness H   the rollout or transcript a call left
  score --dataset F --predictions P [--session S ...] [--out P]   scores, cost and bound
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .dataset import ATTEMPT_LABEL_KEYS, FINE_ROLE_WORDS, ROLE_WORDS
from .labeler_common import BATCH_SIZE, BOUNDARY_TOLERANCE_S, COMMIT_LIMIT, COST_BOUND, PARENT_COMMAND_LIMIT, PREDICTION_KEYS, PROMPT_VERSION, PROMPT_VERSIONS, LabelerError, batches, load_dataset
from .labeler_parse import load_predictions, parse_response, write_predictions
from .labeler_pricing import _UUID_RE, claude_json_result, cost_figures, find_new_session, find_session_files, session_id_from_events
from . import labeler_pricing as _pricing
from .labeler_prompt import PARENT_EDGE_KINDS, _attempt_evidence_result, _attempt_ledger_result, _candidate_parent_result, _candidate_summary, _feat, _prompt_metadata, attempt_evidence, build_prompt, candidate_parent_metadata, candidate_parents, node_view, session_evidence, workflow_evidence
from .labeler_report import report_lines
from .labeler_score import gold_boundaries, score
from .scan import epoch
from .labeler_cli import main


def price_sessions(refs: list[str]) -> list[dict]:
    """Price sessions while preserving facade-level find-session monkeypatching."""
    original = _pricing.find_session_files
    _pricing.find_session_files = find_session_files
    try:
        return _pricing.price_sessions(refs)
    finally:
        _pricing.find_session_files = original


if __name__ == "__main__":
    sys.exit(main())
