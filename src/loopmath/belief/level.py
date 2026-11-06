"""The store level and the benchmark transfer (config `[belief] level`), the default from 0.4.0.

0.3.5's fit learns one success level from every row, and the shipped bundle's sweep rows (loopmath's own test
tasks, 96.6% passing) set it: a fresh store on Terminal-Bench 4 is predicted at 0.633 where agents pass 0.284.
`level = "shared"` restores that fit exactly: `level_settings` then gives `LevelSettings()`, whose `key()` is
empty, so the seed key and every draw are 0.3.5's. Under `level = "store"` (the default, also for a missing or
malformed value):

- every `source:<s>` node is a fixed effect with prior sd `level_sd` (1.5; the cost heads' is below), and
  `fixed:intercept` is held at 0 on the success and gate heads, so each source has its own level and the user's
  level is learned from the user's runs only; `source:user` is in every fitted head, so a store with no runs
  predicts at that wide level instead of the shipped sources' level;
- shipped runs (`sweep`, `e0`, `rq1`, ...) give no success or score rows (`shipped_success = "drop"`);
  their cost and gate rows stay, each under its own source level;
- benchmark success results become one factor per model and effort,
  `S_{m,e} + o_b + t_{b,m} [+ t_{b,m,e}] ~ N(c logit(p), c^2 / (n p (1 - p)) / alpha)`
  (`priors.benchmark_offset_factors`): `o_b` a free offset per benchmark (`bench:<id>`, wide), so a benchmark
  orders models and never sets the level; `t_{b,m}` the model's transfer error on the benchmark (`btx:`, sd
  `c transfer_sd / sqrt(alpha)`, shared by its efforts); `t_{b,m,e}` an effort's own transfer error (`btxe:`, sd
  `c effort_sd / sqrt(alpha)`, key `benchmark_effort_sd`, default 0.5, 0 = none), so an effort gap on a
  benchmark is not taken as exact on the user's tasks; `c` the gap scale (`benchmark_gap_scale`, the inverse of
  the benchmark's slope on the per-task logit scale); `n` the benchmark's runs; `alpha` the power on the
  benchmark likelihood (`benchmark_alpha`, 0 turns them off).

The cost heads (`cost`, and `rate`, the price per token) take the store level too, at their own sd
(`cost_level_sd`, default 0.5), with 0.3.5's free intercept, which carries the shipped runs' level of log dollars
and log dollars per token, so the store's level is its deviation from that one (`fit.make_prior`), or keep
0.3.5's shared level (`cost_level = "shared"`: source nodes and intercept as in 0.3.5, no forced `source:user`),
since run costs carry over between stores far better than pass rates. Gate heads always follow `level`.

Each key can be set on its own; `level = "store"` changes the defaults of `shipped_success` and
`benchmark_factors` to `drop` and `offset`, of `benchmark_effort_sd` to 0.5 (0 under `shared`) and of
`cost_level_sd` to 0.5 (`level_sd` under `shared`). A number that is not finite or out of its range (an sd or
the gap scale at 0 or below, alpha or an sd that may be 0 below it) keeps the default. `key()` holds only the
settings in effect, so keys that `level = "shared"` does not read leave 0.3.5's fit and draws as they are.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

LEVEL_SD = 1.5
INTERCEPT_SD = 1e-3  # `fixed:intercept` under a store level: held at 0, every source's level is its own node
TRANSFER_SD = 0.6
GAP_SCALE = 1.0
EFFORT_SD = 0.5  # `benchmark_effort_sd` under the store level (PHASE-A section 4)
COST_LEVEL_SD = 0.5  # `cost_level_sd` under the store level
USER_LEVEL = "source:user"
COST_HEADS = ("cost", "rate")


@dataclass(frozen=True)
class LevelSettings:  # the field defaults are 0.3.5's fit, `level = "shared"`; level_settings({}) is the store
    level: str = "shared"  # "shared" (0.3.5) | "store"
    level_sd: float = LEVEL_SD
    shipped_success: str = "keep"  # "keep" | "drop"
    benchmarks: str = "reference"  # "reference" (0.3.5's gaps to a reference model) | "offset"
    alpha: float = 1.0
    gap_scale: float = GAP_SCALE
    transfer_sd: float = TRANSFER_SD
    effort_sd: float = 0.0
    cost_level: str = "store"  # "store" | "shared" (0.3.5's level on the cost heads), under `level = "store"`
    cost_level_sd: float = LEVEL_SD

    @property
    def store(self) -> bool:
        return self.level == "store"

    @property
    def own_success(self) -> bool:
        return self.shipped_success == "drop"

    def for_head(self, name: str) -> "LevelSettings | None":
        """The settings one head is fitted under: None (0.3.5's prior) for a cost head under a shared cost level."""
        if name not in COST_HEADS or not self.store:
            return self
        return None if self.cost_level == "shared" else replace(self, level_sd=self.cost_level_sd)

    def key(self) -> dict:
        """The settings in effect, for the seed key and meta.json; empty when they are 0.3.5's (`shared`, `keep`,
        `reference`), whatever the keys those leave unread."""
        return {} if self._in_effect() == LevelSettings()._in_effect() else self._in_effect()

    def _in_effect(self) -> dict:
        fields = asdict(self)
        names = ["level", "shipped_success", "benchmarks"]
        if self.store:
            names += ["level_sd", "cost_level"] + (["cost_level_sd"] if self.cost_level == "store" else [])
        if self.benchmarks == "offset":
            names += ["alpha"] + (["gap_scale", "transfer_sd", "effort_sd"] if self.alpha > 0 else [])
        return {n: fields[n] for n in names}


def _num(value, default: float, low: float = 0.0, *, above: bool = False) -> float:
    """A finite number of at least `low` (above it with `above`), else `default`; an int past a float's range is
    malformed too."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not _finite(value):
        return default
    return float(value) if (value > low if above else value >= low) else default


def _finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def level_settings(config: dict) -> LevelSettings:
    """The `[belief]` keys of a config.toml table; unknown or malformed values keep the default."""
    b = config.get("belief") if isinstance(config.get("belief"), dict) else {}
    level = b.get("level") if b.get("level") in ("shared", "store") else "store"
    store = level == "store"
    level_sd = _num(b.get("level_sd"), LEVEL_SD, above=True)
    shipped = b.get("shipped_success") if b.get("shipped_success") in ("keep", "drop") else (
        "drop" if store else "keep")
    bench = b.get("benchmark_factors") if b.get("benchmark_factors") in ("reference", "offset") else (
        "offset" if store else "reference")
    return LevelSettings(level=level, level_sd=level_sd,
                         shipped_success=shipped, benchmarks=bench,
                         alpha=_num(b.get("benchmark_alpha"), 1.0),
                         gap_scale=_num(b.get("benchmark_gap_scale"), GAP_SCALE, above=True),
                         transfer_sd=_num(b.get("transfer_sd"), TRANSFER_SD),
                         effort_sd=_num(b.get("benchmark_effort_sd"), EFFORT_SD if store else 0.0),
                         cost_level=b.get("cost_level") if b.get("cost_level") in ("store", "shared") else "store",
                         cost_level_sd=_num(b.get("cost_level_sd"), COST_LEVEL_SD if store else level_sd, above=True))


def shipped(source: str) -> bool:
    """A source of the shipped bundle: not the user's store and not a shared import."""
    return source != "user" and not source.startswith("shared")
