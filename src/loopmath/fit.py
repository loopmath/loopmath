"""loopmath.fit -- E1 v2 surface model wiring (SPEC section 7, CP3).

Purpose: let the user run E1a/E1b on the existing sweep data with zero plumbing
work. This module assembles the model table from the sweep run records,
implements the observe/holdout/reveal masking grammar, builds the PyMC model
skeleton, and scores held-out cells.

Hard constraints this module honors (SPEC section 0 + section 7, 08-31):
  - No network calls; reads only from the local sweep directory.
  - DESCRIPTIVE ONLY: nothing here computes or returns a pass/fail verdict,
    a gate, or a threshold comparison. `transfer_test` reports numbers.
  - No harness anchor: the model never gets a claude-vs-openai family term.
  - `loopmath.scoring` (owned by another module) is imported lazily inside
    `transfer_test`, the only function that needs it, so this module imports
    cleanly whether or not that module exists yet. `loopmath.price` and
    `loopmath.ingest.base` are frozen and stable, so they are imported at module
    load time like any other dependency.
  - Needs the [bayes] extra (pymc, arviz). `assemble_table`/`parse_mask`/
    `apply_mask` do not need it; `build_model`/`sample` do and say so via
    `require_bayes`.

Fix-round changes (this pass), by decision:

DATA DECISION A -- pricing (AMENDED, see below). Every DEV attempt in the
sweep carries `ext.tokens` (four fields: input, output, cache_read,
cache_write); only the Claude-family ones also carry a recorded
`ext.cost_usd`. Pricing GPT rows from tokens and Claude rows from a recorded
dollar figure would mix two pricing regimes into one model and manufacture a
fake cross-family cost effect. Under the amended (four-stream) price table
this is no longer a hedge against disagreement: the table itself is derived
from real billing data, so on the rows that do carry a recorded cost, the
token-derived figure reproduces it (see `_cost_price_note`, an exact-match
count and worst residual computed at runtime from whatever rows carry both,
never hardcoded). Pricing uniformly from tokens is still the right call --
GPT rows never carry a recorded cost at all, so a uniform method is the only
one that treats every row the same way, and it is now demonstrably the same
number as the recorded one wherever both exist, rather than merely the more
defensible guess. So `usd` is computed uniformly, for every row, from that
row's own developer-attempt token counts against the packaged price table
(`loopmath.price.load_prices` / `price_run`), mapping the sweep's four token
fields one-to-one onto the four-stream contract (see the SPEC AMENDMENT
paragraph below): `in = input`, `cache_read = cache_read`, `cache_write =
cache_write`, `out = output`. The recorded figure survives as its own
column, `usd_recorded` (None when absent), and is never mixed into `usd`.
`usd_todo_rate` marks rows priced from a placeholder rate. A row whose
tokens cannot be priced is excluded and counted by name in `dropped`, never
priced at zero. `run_fit` prints the resulting honesty note
(`assembly["cost_price_note"]`) directly, because `cli.py`'s `fit_verb` does
not forward it (see the cli.py note below).

SPEC AMENDMENT (Analyst, mid fix-round, overrides the original text of
DATA DECISION A above): the record and price table are four streams, not
three -- `in`, `cache_read`, `cache_write`, `out` -- because a cache write
and a cache read are priced very differently (on Anthropic a cache write can
run up to 20x a cache read; OpenAI charges nothing for either). The earlier
plan folded cache writes into `in`; that silently overcharged one provider
and undercharged the other, which is exactly the kind of manufactured,
unequal error DATA DECISION A itself was written to rule out. So the mapping
above is direct, nothing is summed on the way in, and `loopmath.price.price_run`
(owned by another module, updated in parallel to the same four streams) is
called unchanged. A related, separate honesty gap: GPT-5.6 has a long-context
surcharge above roughly 272k tokens per request that cannot be resolved from
a per-attempt record (which side of the threshold a given request's context
fell on is not recorded). `_long_context_note` does not try to guess this
per row -- an earlier version of this function counted rows whose own total
token count crossed 272k, which is exactly the false precision this caveat
exists to avoid: a request's context size is not the same thing as an
attempt's summed tokens across however many calls it made, so that count
implied knowledge the data does not actually support. It now reuses
`loopmath.price.price_all` / `loopmath.price.warning_lines` directly, over this
table's own developer-attempt tokens, and takes their NOTE line verbatim
(`assembly["long_context_note"]`, printed by `run_fit`), so the caveat is
worded once, by the module that owns the price table, not restated here
with a guess this module cannot verify.

DATA DECISION B -- which files are the R1 slice. SPEC section 7 says "R1
variant, t1-t5, t7"; direct inspection of the 660-file corpus shows t7 tasks
exist under exactly one variant, `plan-gpt-5.6-luna-low@rev-claude-opus-5-
xhigh` (t7 does not exist under the plain `rev-claude-opus-5-xhigh`
variant), while t1-t5 exist under the plain variant. So: t1-t5 come from the
plain variant, t7-* come from the pinned-planner variant, and t1-t5 are
*not* also pulled from the other planner-pinned variant that also carries
them (`plan-gpt-5.6-sol-xhigh@...`) -- that would double the rows and
confound planner with task. `planner` ("none" or "gpt-5.6-luna-low") and
`variant` (the full variant string) columns make this visible instead of
folding it into the task effect. Target: 300 rows (150 + 150). Actual: 300
files selected, 277 rows assembled -- 23 t7 rows are excluded under
`dropped["dev_zero_tokens"]` (a verified data condition: every developer
attempt in those rows reports all-zero tokens, priced correctly to $0.00,
which is undefined for `log10(usd)` and correlates heavily with rejection;
see `assemble_table`'s docstring and the fix-round report). Every other drop
reason is 0 on this snapshot of the corpus.

FLAG 1 (originally "partial uphold", REVERSED in the correction pass below)
-- this module used to keep `usd` (the developer attempt's own cost) as the
modelled response and carry `usd_run_total` only as an unused diagnostic
column, on the reasoning that the model in SPEC section 7 is indexed by the
*developer's* model and effort and a reviewer cost that is the same
claude-opus-5-xhigh in every cell would inject a near-constant offset not
attributable to the model being estimated. Two external reviewers correctly
called this out: SPEC section 7 asks for per-run USD for the whole session
set, planner and reviewer sessions included, and `usd` alone silently drops
that cost from every printed and modelled number. See the CORRECTION
paragraph immediately below for what changed and why it was safe to change.

CORRECTION (this pass) -- measured first, then decided: across the 277
assembled rows, `usd_run_total` is non-null for all 277 (0 nulled; see
`assembly["usd_run_total_nulled"]`), so switching the modelled response does
not silently shrink the table. `build_model`'s cost head, `transfer_test`'s
MLPD/coverage/Kendall-tau/within-factor/decision-regret scoring, and every
printed line describing the outcome now use `usd_run_total` -- "the whole
session set for the run, planner and reviewer included" -- not `usd`.
`usd` (the developer attempt's own cost) is kept as an auditable diagnostic
column, still computed and still described by `_cost_price_note`, so the
change can be checked against the old figure; it is no longer what the
StudentT head fits or what the transfer test scores against. `run_fit`
still prints the median share of `usd_run_total` that the developer attempt
represents, so a reader can see how much of the modelled cost the developer
attempt itself accounts for.

PATCH 4 (disclosed, not structurally fixed) -- the 23 `dev_zero_tokens` rows
(see DATA DECISION B) are dropped from the table before the acceptance head
ever sees them, and that drop is not outcome-neutral: those rows skew
rejection-heavy compared to the corpus (see `_acceptance_bias_note`'s
runtime measurement, printed by `run_fit`), so removing them inflates the
modelled acceptance rate. Keeping them in the Bernoulli likelihood while
excluding them only from the StudentT cost likelihood would need
`build_model` to accept two different row sets sharing one set of
model/effort/task coords, which today's single-`df` model skeleton does not
support without touching the masking and `predict`/`transfer_test` scoring
path as well -- a larger change than this fix round's time budget allows for
a 23-row effect. So this pass disclosed the bias, measured at runtime and
printed by `run_fit`, rather than restructuring the model to remove it.

FLAG 2 (uphold) -- resolution A already stops rows being dropped for a
missing *recorded* cost. What is left: a row whose developer attempt(s)
cannot be priced from tokens is an explicitly counted exclusion
(`dropped["dev_tokens_unusable"]` / `dropped["dev_model_unpriced"]`), never
a partial sum. The same rule applies to `usd_run_total`: if any attempt in
the run is unpriceable, `usd_run_total` is `None` for that row and counted
in `assembly["usd_run_total_nulled"]`, never a partial sum presented as a
total.

FLAG 3 (uphold) -- `_validate_accepted` accepts only a real `bool`, the
exact lowercased strings `"true"`/`"false"`, or the ints `0`/`1`. Anything
else, including `None`, is an excluded row counted under
`dropped["invalid_accepted"]`. No guessing (`bool()` on a non-empty string
or on `None` is exactly the bug this replaces).

FLAG 4 (uphold, proven) -- resolutions A and B together mean the default
E1a mask (`luna:low,medium,xhigh;sol:medium;terra:medium`, all GPT-family)
now observes real rows: it no longer needs a Claude-only table.
`test_default_e1a_mask_observes_and_holds_out_real_rows` in `test_fit.py`
runs the default mask against the real assembled table and asserts positive
observed and held-out counts (skipping, not vacuously passing, if the
sweep directory is unavailable in the environment running the test).

FLAG 5 (uphold, most important) -- `predict` no longer fixes an unseen
model's or task's random effect at zero. `beta_m`, `task_effect`, and
`task_gamma` each have a fitted hyperprior scale (`tau_beta`, `tau_task`,
`tau_task_g`); an unseen level draws its effect from `Normal(0, that fitted
scale)`, per predictive sample, so the predictive spread carries the
uncertainty of a level the fit never saw. `alpha_m` and `gamma_m` have no
*fitted* scale of their own (their prior scale, 1.5, is a fixed
hyperparameter in `build_model`, not estimated from data); an unseen model
draws those two from that same fixed scale, which is still the population
distribution the model assumes new models come from, just not itself
learned. One shared draw per unique unseen name (not per row), so a level
that recurs across several held-out rows is treated consistently, the way
it would be if it had actually been observed. `transfer_test` counts and
prints how many held-out rows needed an unseen-level draw, and names the
models/tasks involved, because a reader needs to know which numbers rest on
a level the model never saw.

FLAG 6 (uphold) -- `test_assemble_table_real_sweep_data` now pins the exact
row count (277, after the `dev_zero_tokens` exclusion above), the exact
model and effort sets, the exact drop counts, and asserts every (model,
effort) cell is nonempty (with each cell's live count printed for a human to
check), instead of a `groupby().size() >= 0` check that could never fail.

A gap found while wiring this up, not asked for in the flags: `cli.py`'s
`fit_verb`/`transfer_test_verb` (owned by the lead, out of scope for this
fix) read `result["n_rows"]`, `result["sweep_dir"]`, `result["n_observed"]`,
`result["n_heldout"]`, `result["mask_spec"]` from `run_fit`'s return value,
and `loopmath.scoring.summarize` reads a `"within_factor"` key -- neither
existed before this pass (the old keys were nested under `"shapes"`, and the
old key was `"within_2x"`). `loopmath fit` and `loopmath transfer-test` could not
run end to end without these; `run_fit`/`transfer_test` now populate both
the old and the cli.py-expected flat keys. Also as a direct consequence:
`run_fit` and `transfer_test` print a small number of mandatory, loud,
data-honesty lines directly (the cost/planner notes, the median run-total
share, the unseen-level count) because `cli.py` does not yet forward the
notes these functions already return in their result dicts. Every other
line of terminal output is still the CLI's job; this is a narrow, named
exception, not a reversal of that design.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from .ingest.base import canonical_model
from .price import PriceTable, load_prices, price_all, price_run, warning_lines
from . import fit_assembly as _assembly
from . import fit_bayes as _bayes
from . import fit_masks as _masks
from . import fit_predict as _predict
from . import fit_pricing as _pricing
from . import fit_transfer as _transfer
from .fit_assembly import *
from .fit_bayes import *
from .fit_masks import *
from .fit_predict import *
from .fit_pricing import *
from .fit_transfer import *


_SUPPORT_REBINDINGS = {
    "BAYES_HINT": (_bayes,),
    "DEFAULT_SWEEP_DIR": (_assembly, _transfer),
    "EFFORT_ORDER": (_bayes, _masks),
    "Mask": (_masks,),
    "MODEL_SHORT": (_masks,),
    "Path": (_assembly, _transfer),
    "PriceTable": (_assembly, _pricing),
    "SEED": (_transfer,),
    "TRANSFER_TEST_NOTE": (_transfer,),
    "_ATTEMPT_MODEL_PREFIX": (_pricing,),
    "_FULL_MODEL_NAMES": (_masks,),
    "_SWEEP_TOKEN_FIELDS": (_pricing,),
    "_TABLE_COLUMNS": (_assembly,),
    "_TASK_PLANNER": (_assembly,),
    "_TASK_VARIANT": (_assembly,),
    "_acceptance_bias_note": (_assembly,),
    "_attempt_model_name": (_pricing,),
    "_cost_price_note": (_assembly,),
    "_long_context_note": (_assembly,),
    "_map_sweep_tokens": (_assembly, _pricing),
    "_parse_cell_groups": (_masks,),
    "_parse_reveal_groups": (_masks,),
    "_parse_sweep_filename": (_assembly,),
    "_price_attempt_tokens": (_assembly, _pricing),
    "_price_run_total": (_assembly, _pricing),
    "_resolve_model_name": (_masks,),
    "_validate_accepted": (_assembly,),
    "apply_mask": (_transfer,),
    "assemble_table": (_transfer,),
    "build_model": (_bayes,),
    "canonical_model": (_assembly, _pricing),
    "json": (_assembly,),
    "load_prices": (_assembly,),
    "np": (_bayes, _masks, _predict, _pricing, _transfer),
    "parse_mask": (_transfer,),
    "pd": (_assembly, _bayes, _masks, _predict, _pricing, _transfer),
    "predict": (_transfer,),
    "price_all": (_pricing,),
    "price_run": (_pricing,),
    "require_bayes": (_bayes,),
    "sample": (_transfer,),
    "transfer_test": (_transfer,),
    "warning_lines": (_pricing,),
}


class _FitModule(ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _SUPPORT_REBINDINGS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _FitModule
