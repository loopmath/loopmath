"""Reported quality per scale, new-run draws and the value function (numpy and scipy only, no engine).

Fitting scale. Every dimension is fitted on a scale where higher is better:

- `linear`, `fraction`: the range mapped linearly to [0, 1].
- `levels`: level number l of L at (l - 1) / (L - 1); fixed cut points halfway between neighbours.
- `binary`, `items`: probit units.

A component with `better: lower` is turned over when a raw value is mapped (`fitting_value`,
`fitting_count`), so arm means, reported quality, weights and gates always read "higher is better".
`own_scale` turns a reported quality back to the component's own scale.

Reported quality `m(q)` lies in [0, 1]: `q` clipped for `linear`, `fraction` and `levels`;
1 when `q >= 0` for `binary`; `Phi(q)` for `items`.

A dimension is a dict with `scale` and `better`, as `contracts.dimensions` and a fit file give it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr

from .contracts import FINAL, ContractError, dimension_id

UNIT_KINDS = ("linear", "fraction", "levels")  # fitted on [0, 1]
PROBIT_KINDS = ("binary", "items")             # fitted in probit units


def fixed_cuts(n_levels: int) -> np.ndarray:
    """The n_levels - 1 fixed cut points of a `levels` scale on [0, 1], halfway between neighbouring levels."""
    return (np.arange(n_levels - 1) + 0.5) / (n_levels - 1)


def level_index(scale: dict, value: Any) -> int:
    """The 0-based position of a level in `scale.levels`. ValueError when it is not a level."""
    for n, level in enumerate(scale["levels"]):
        if type(level) is type(value) and level == value:
            return n
    raise ValueError(f"{value!r} is not a level")


def fitting_value(dim: dict, value: Any) -> float:
    """A raw `linear`, `fraction`, `levels` or `binary` value on the fitting scale's observed side.

    `linear`, `fraction` and `levels` give a number in [0, 1]; `binary` gives 0.0 or 1.0. `better: lower`
    turns the value over."""
    scale = dim["scale"]
    kind = scale["kind"]
    if kind == "linear":
        low, high = scale["range"]
        x = (float(value) - low) / (high - low)
    elif kind == "levels":
        x = level_index(scale, value) / (len(scale["levels"]) - 1)
    elif kind in ("fraction", "binary"):
        x = float(value)
    else:
        raise ValueError("an items reading is a count: use fitting_count")
    return 1.0 - x if dim.get("better", "higher") == "lower" else x


def fitting_level(dim: dict, value: Any) -> int:
    """The 0-based level of a `levels` reading after orientation: 0 is the worst level."""
    scale = dim["scale"]
    n = level_index(scale, value)
    return len(scale["levels"]) - 1 - n if dim.get("better", "higher") == "lower" else n


def fitting_count(dim: dict, passed: int, of: int) -> tuple[int, int]:
    """An `items` count (k passed of n) after orientation."""
    return (of - passed, of) if dim.get("better", "higher") == "lower" else (passed, of)


def reported(q: Any, kind: str) -> np.ndarray:
    """Reported quality `m(q)` of one run for one scale kind, elementwise."""
    q = np.asarray(q, dtype=float)
    if kind in UNIT_KINDS:
        return np.clip(q, 0.0, 1.0)
    if kind == "binary":
        return np.where(np.isnan(q), np.nan, (q >= 0).astype(float))
    if kind == "items":
        return ndtr(q)
    raise ValueError(f"unknown scale kind {kind!r}")


def reported_all(q: Any, kinds: Sequence[str]) -> np.ndarray:
    """Reported quality of `q[..., D]`, with one scale kind per entry of the last axis."""
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    for d, kind in enumerate(kinds):
        out[..., d] = reported(q[..., d], kind)
    return out


def arm_reported(mu: Any, sd: Any, kind: str) -> np.ndarray:
    """An arm's reported quality: the mean of `m(q)` over a new input and a new run, `q ~ N(mu, sd^2)`.

    `sd` is the total sd, `sqrt(sd_input^2 + sd_run^2)`. `binary` gives the pass rate `Phi(mu / sd)`,
    `items` the mean share `Phi(mu / sqrt(1 + sd^2))`, the others the mean of `q` clipped to [0, 1]."""
    mu, sd = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(sd, dtype=float))
    safe = np.where(sd > 0, sd, 1.0)
    if kind == "binary":
        return np.where(sd > 0, ndtr(mu / safe), np.where(np.isnan(mu), np.nan, (mu >= 0).astype(float)))
    if kind == "items":
        return ndtr(mu / np.sqrt(1.0 + sd * sd))
    if kind in UNIT_KINDS:
        a, b = (0.0 - mu) / safe, (1.0 - mu) / safe
        pdf = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)  # noqa: E731
        inside = mu * (ndtr(b) - ndtr(a)) + safe * (pdf(a) - pdf(b)) + (1.0 - ndtr(b))
        return np.where(sd > 0, inside, np.clip(mu, 0.0, 1.0))
    raise ValueError(f"unknown scale kind {kind!r}")


def own_scale(m: Any, dim: dict) -> np.ndarray | None:
    """A reported quality on the component's own scale, or None when the scale has none.

    `linear` gives a number in the range, `fraction` the share, `levels` a level number from 1 to L (not
    rounded). `better: lower` is turned back, so the result reads as the user wrote the scale."""
    scale = dim["scale"]
    kind = scale["kind"]
    if kind not in UNIT_KINDS:
        return None
    m = np.asarray(m, dtype=float)
    if dim.get("better", "higher") == "lower":
        m = 1.0 - m
    if kind == "linear":
        low, high = scale["range"]
        return low + m * (high - low)
    if kind == "levels":
        return 1.0 + m * (len(scale["levels"]) - 1)
    return m


def covariance(sd: Any, corr: Any) -> np.ndarray:
    """Covariance matrices `[..., D, D]` from sds `[..., D]` and correlation matrices `[..., D, D]`."""
    sd = np.asarray(sd, dtype=float)
    return sd[..., :, None] * np.asarray(corr, dtype=float) * sd[..., None, :]


def _root(cov: np.ndarray) -> np.ndarray:
    """A matrix square root per draw that also takes a singular covariance (a factor structure)."""
    w, v = np.linalg.eigh(cov)
    return v * np.sqrt(np.clip(w, 0.0, None))[..., None, :]


def new_runs(draws: Any, rng: np.random.Generator, n: int = 1, arms: Sequence[int] | None = None) -> np.ndarray:
    """Hidden quality of new runs on new inputs, per posterior draw: `[S, n, A, D]`.

    For each draw and each of `n` new inputs, one input effect `v ~ N(0, Sigma_input)` is shared by every
    arm (so arms are paired on inputs) and each arm gets its own run effect `u ~ N(0, Sigma_run)`:
    `q = mu_a + v + u`. `draws` is a mapping with `mu`, `sd_input`, `sd_run`, `corr_input`, `corr_run`
    (a `Fit.draws` mapping). `arms` keeps a subset of arm positions."""
    mu = np.asarray(draws["mu"], dtype=float)
    if arms is not None:
        mu = mu[:, list(arms), :]
    S, A, D = mu.shape
    root_input = _root(covariance(draws["sd_input"], draws["corr_input"]))
    root_run = _root(covariance(draws["sd_run"], draws["corr_run"]))
    v = np.einsum("sde,sne->snd", root_input, rng.standard_normal((S, n, D)))
    u = np.einsum("sde,snae->snad", root_run, rng.standard_normal((S, n, A, D)))
    return mu[:, None, :, :] + v[:, :, None, :] + u


@dataclass
class ValueSpec:
    """A value file compiled against the dimensions of a fit or a rubric.

    `weights[D]` is 0 outside the weighted final dimensions. Each gate is `(d, threshold)` on reported
    quality: the gate passes when `m[d] >= threshold`."""

    weights: np.ndarray
    gates: list[tuple[int, float]]
    gated_value: float
    objective: dict
    kinds: list[str]
    gate_labels: list[str] = field(default_factory=list)

    def passes_each(self, m: Any) -> np.ndarray:
        """Per gate, whether `m[..., D]` passes: `[..., G]`."""
        m = np.asarray(m, dtype=float)
        if not self.gates:
            return np.ones(m.shape[:-1] + (0,), dtype=bool)
        return np.stack([m[..., d] >= t for d, t in self.gates], axis=-1)

    def passes(self, m: Any) -> np.ndarray:
        """Whether `m[..., D]` passes every gate."""
        return self.passes_each(m).all(axis=-1)

    def weighted(self, m: Any) -> np.ndarray:
        """`w'm`, over the weighted dimensions only (an unweighted dimension may hold NaN)."""
        m = np.asarray(m, dtype=float)
        used = np.flatnonzero(self.weights)
        return m[..., used] @ self.weights[used]

    def value(self, m: Any) -> np.ndarray:
        """`V = pass x w'm + (1 - pass) x gated_value` for reported quality `m[..., D]`."""
        return np.where(self.passes(m), self.weighted(m), self.gated_value)

    def value_of_q(self, q: Any) -> np.ndarray:
        """`V` for hidden quality `q[..., D]`."""
        return self.value(reported_all(q, self.kinds))


def gate_threshold(dim: dict, at_least: Any) -> float:
    """A gate's `at_least`, given on the component's own scale, as a threshold on reported quality.

    `levels`: a level. `linear`: a number in the range. `fraction`, `items`: a share from 0 to 1.
    `binary`: `at_least` is None and the gate is "passes". For a `better: lower` component `at_least` is
    still read on the own scale and means "this value or better", so 2 on a 0 to 10 lower-is-better
    scale passes values of 2 and below. Raises ValueError with the reason when it does not fit."""
    scale = dim["scale"]
    kind = scale["kind"]
    lower = dim.get("better", "higher") == "lower"
    if kind == "binary":
        if at_least is not None:
            raise ValueError("a binary gate has no at_least: the gate is \"passes\"")
        return 1.0
    if at_least is None:
        raise ValueError(f"a gate on a {kind} component needs at_least")
    if kind == "levels":
        try:
            x = level_index(scale, at_least) / (len(scale["levels"]) - 1)
        except ValueError:
            raise ValueError(f"{at_least!r} is not one of the levels "
                             f"{', '.join(str(level) for level in scale['levels'])}") from None
    else:
        low, high = scale["range"] if kind == "linear" else (0.0, 1.0)
        if isinstance(at_least, str) or not low <= at_least <= high:
            raise ValueError(f"{at_least!r} is outside {low:g} to {high:g}")
        x = (float(at_least) - low) / (high - low)
    return 1.0 - x if lower else x


def compile_value(value: dict, dims: Sequence[dict], file: str = "value.json") -> ValueSpec:
    """Check a normalised value file against dimensions and compile it. Refusals are `ContractError`s.

    `dims` is `contracts.dimensions(rubric)` or the `dimensions` member of a fit; an entry with
    `"relative": true` cannot carry a gate. Refused, each by path: a weight or gate on an unknown
    component, a threshold off the component's scale, a gate on a relative dimension, and a weight on
    `overall` mixed with weights on criteria."""
    by_id = {dim["id"]: n for n, dim in enumerate(dims)}
    weights = np.zeros(len(dims))
    roles = set()
    for name, weight in value["weights"].items():
        d = by_id.get(dimension_id(FINAL, name))
        if d is None:
            raise ContractError(file, f"weights.{name}", f'unknown final component "{name}"',
                                "weight a component that applies to the final step", "unknown_component")
        weights[d] = weight
        if weight:
            roles.add(dims[d].get("role", "criterion"))
    if roles == {"overall", "criterion"}:
        raise ContractError(file, "weights", "a weight on the overall component is mixed with weights on criteria",
                            "weight the overall component alone, or the criteria alone", "overall_mixed")
    gates, labels = [], []
    for n, gate in enumerate(value["gates"]):
        key = dimension_id(gate["step"], gate["component"])
        d = by_id.get(key)
        if d is None:
            raise ContractError(file, f"gates[{n}]", f'unknown dimension "{key}"',
                                "name a component and a step it applies to", "unknown_component")
        if dims[d].get("relative"):
            raise ContractError(file, f"gates[{n}]", f'the dimension "{key}" is relative (pairs only)',
                                "gate a dimension that has scores, or drop the gate", "relative_gate")
        try:
            threshold = gate_threshold(dims[d], gate.get("at_least"))
        except ValueError as exc:
            raise ContractError(file, f"gates[{n}].at_least", str(exc),
                                "give the threshold on the component's own scale", "bad_threshold") from None
        gates.append((d, threshold))
        labels.append(key)
    return ValueSpec(weights, gates, float(value["gated_value"]), dict(value["objective"]),
                     [dim["scale"]["kind"] for dim in dims], labels)
