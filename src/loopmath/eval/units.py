"""The unit of the draws on `binary` and pairs-only dimensions: the anchors' total noise of one reading.

A `binary` reading and a pair verdict show only the sign of a latent sum, so the data fix the latent
scale of such a dimension only up to a factor. The engines fit it in the anchors' call noise: on a
`binary` dimension with readings the anchors' mean log `sigma` is 0, on a dimension with pairs only the
anchors' mean log `sigma_pair` is 0. That unit is weakly identified, because the data pin an anchor's
total noise of one reading and split it poorly into its judge-by-output part `tau` and its call noise
`sigma`. Every latent quantity on the dimension moves with that split. The fit file therefore reports
these dimensions in the anchors' total noise of one reading, which the data do pin. It is a change of
unit for the latent quantities on the dimension, not of the model or the fit: pass rates and every
other reported quality stay as they were.

The rule, per draw, for dimension d:

- `binary` with readings: `s_d = exp(sum_j w_jd log sqrt(sigma_jd^2 + tau_jd^2 + sd_judge_input_jd^2))`.
  `w` is the engine's own unit rule: 1/n over the n anchors who read d, else over every judge who read d.
  `sd_judge_input` counts where the judge-by-input term is on, else it is 0. With one anchor whose
  call noise is the engine's unit, `s_d = sqrt(1 + tau^2 + sd_judge_input^2)`.
- pairs only: `s_d = exp(sum_j w_jd log sqrt(sigma_pair_jd^2 + 2 tau_jd^2))`, `w` over the anchors who
  compared on d, else every judge who did. `sqrt(sigma_pair^2 + 2 tau^2)` is the sd of the noise of
  one pair verdict: the judge-by-output terms of the two outputs plus the call noise. Bias and the
  judge-by-input term cancel in a pair of outputs of one input. A gap of 1 between two outputs is then
  picked right with probability `Phi(1)`, before ties and position bias.
- every other dimension: `s_d = 1`. `linear`, `fraction` and `levels` dimensions are fitted on [0, 1]
  (with fixed cuts for `levels`) and `items` dimensions in the item model's unit; the data fix those.

On a dimension with `s_d != 1`, `CONVERTED` arrays are divided by `s_d`: arm means (and so arm
differences), run quality, `sd_input`, `sd_run`, bias, family effects, `tau`, `sigma`,
`sd_judge_input` and `sigma_pair`. Correlations and halo shares have no unit. `position` and
`tie_width` are one number per judge, shared by every dimension the judge compares on, so they stay in
the engine's unit; a reader puts them on a dimension's unit with the `unit` draws, `s_d` per draw.

After the division the anchors' mean log total noise is 0 on each such dimension, so `factor` of the
converted draws is 1 in every draw. The anchor's call noise was fixed at 1 in the engine's unit; in
the reported unit it is `1 / s_d` and uncertain.

Probabilities read per draw do not move: pass rates, the judges' false pass, false fail, flip and
agreement, r2, reliability, tie rates and picks. A figure averaged over draws in latent units does
move beyond the change of unit, because `s_d` differs from draw to draw, and so would a choice made on
it. A latent figure inside a nonlinear map moves too: a value weight on a relative criterion clips
the latent quality itself, and the eigenvalue root of a singular covariance turns when one
dimension's unit changes.

So every decision a reader makes is made in the unit the engine fitted in (`in_fitted_unit`), and the
reported unit is for the figures shown: a fit of schema 2 and the same draws in a fit of schema 1 give
the same verdicts, choices, flags, ranks and picks. That covers the value report and the steps hint
(`value.py`, `steps.py`), the judges' choices, the target error and the second judge (`judges.py`,
whose table still shows bias, tau, sigma and mse in the reported unit), the rubric's health
(`rubric.py`, whose latent figures are in the fitted unit), the active label scores (`labels.py`)
and the simulator's `overall` and `value` truths (`simulate.py`).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

CONVERTED = ("mu", "q_run", "sd_input", "sd_run", "bias", "tau", "sigma", "family", "sd_judge_input",
             "sigma_pair")
# The unit of a dimension's draws, and the unit its engine fitted it in.
UNITS = ("total_noise", "total_pair_noise", "scale", "items")
FITTED_IN = {"total_noise": "call_noise", "total_pair_noise": "pair_noise", "scale": "scale", "items": "items"}
OLD_UNITS = ("call_noise", "pair_noise", "scale", "items")   # what a fit of schema 1 holds
RECORD = ("unit", "fitted_in", "judges", "factor")


def unit_of(kind: str, relative: bool) -> str:
    """The reported unit of a dimension from its scale kind and whether it has pairs only."""
    if relative:
        return "total_pair_noise"
    if kind == "binary":
        return "total_noise"
    return "items" if kind == "items" else "scale"


def old_unit_of(kind: str, relative: bool) -> str:
    """The unit of a dimension in a fit of schema 1: the engine's unit."""
    return FITTED_IN[unit_of(kind, relative)]


def factor(draws: Mapping[str, Any], sigma_w: Any, pair_w: Any, binary: Any, pairs_only: Any,
           ji_on: Any = None) -> np.ndarray:
    """`s[..., D]`: the size of one reported unit in the draws' unit, per draw.

    `draws` holds `sigma`, `tau`, `sd_judge_input` and `sigma_pair` with shape `[..., J, D]` (any leading
    axes, or none). `sigma_w` and `pair_w` are `[J, D]` weights that sum to 1 over the judges who carry
    the unit; `binary` and `pairs_only` are `[D]` masks of the dimensions under each rule (a dimension
    under neither gets 1). `ji_on[J, D]` says where the judge-by-input term is on; where it is off, or
    not given, its sd counts as 0. Values of judges with weight 0 are not read, NaN or not."""
    sigma = np.asarray(draws["sigma"], dtype=float)
    tau = np.asarray(draws["tau"], dtype=float)
    pair = np.asarray(draws["sigma_pair"], dtype=float)
    ji = np.asarray(draws["sd_judge_input"], dtype=float)
    sigma_w, pair_w = np.asarray(sigma_w, dtype=float), np.asarray(pair_w, dtype=float)
    binary, pairs_only = np.asarray(binary, dtype=bool), np.asarray(pairs_only, dtype=bool)
    on = np.ones(sigma.shape[-2:], dtype=bool) if ji_on is None else np.asarray(ji_on, dtype=bool)
    ji = np.where(on, ji, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        one = 0.5 * np.log(sigma ** 2 + tau ** 2 + ji ** 2)
        two = 0.5 * np.log(pair ** 2 + 2.0 * tau ** 2)
    log_s = np.where(sigma_w > 0, one, 0.0) * sigma_w
    log_p = np.where(pair_w > 0, two, 0.0) * pair_w
    log_s, log_p = log_s.sum(axis=-2), log_p.sum(axis=-2)                       # [..., D]
    out = np.where(binary & ~pairs_only, np.exp(log_s), 1.0)
    return np.where(pairs_only, np.exp(log_p), out)


def convert(draws: Mapping[str, Any], unit: Any) -> dict[str, np.ndarray]:
    """The draws with every `CONVERTED` array divided by `unit[..., D]` on its dimension axis (the last),
    and `unit` added. Other arrays are kept as they are."""
    unit = np.asarray(unit, dtype=float)
    out = {name: np.asarray(x) for name, x in draws.items()}
    lead = unit.shape[:-1]
    for name in CONVERTED:
        if name not in out:
            continue
        x = np.asarray(out[name], dtype=float)
        extra = x.ndim - unit.ndim
        out[name] = x / unit.reshape(lead + (1,) * extra + unit.shape[-1:])
    out["unit"] = unit
    return out


def in_fitted_unit(draws: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The draws of a fit file put back in the unit the engine fitted them in: every `CONVERTED` array times
    the `unit` draws on its dimension axis, and `unit` set to 1. A fit without `unit` draws (schema 1, or
    `Fit.draws` filling 1) comes back as it is. For readers whose choices should not move with the unit."""
    out = {name: np.asarray(x) for name, x in draws.items()}
    if "unit" not in out:
        return out
    unit = np.asarray(out["unit"], dtype=float)
    lead = unit.shape[:-1]
    for name in CONVERTED:
        if name in out:
            x = np.asarray(out[name], dtype=float)
            out[name] = x * unit.reshape(lead + (1,) * (x.ndim - unit.ndim) + unit.shape[-1:])
    out["unit"] = np.ones_like(unit)
    return out


def for_design(design: Any, draws: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The engine's draws of a design (`model.Design`) in the reported unit, with the `unit` draws."""
    data = design.data
    s = factor(draws, data["sigma_w"], data["pair_w"], data["sigma_fixed"], data["pairs_only"],
               design.masks["ji_on"])
    return convert(draws, s)


def unit_judges(weights: Any, judges: Sequence[str]) -> list[list[str]]:
    """Per dimension, the judges with a weight in `weights[J, D]`."""
    weights = np.asarray(weights, dtype=float)
    return [[judges[j] for j in range(weights.shape[0]) if weights[j, d] > 0] for d in range(weights.shape[1])]


def records(dims: Sequence[dict], relative: Sequence[str], who: Sequence[Sequence[str] | None] | None = None,
            unit: Any = None) -> list[dict]:
    """Per dimension, the `unit` record of the fit file: `{"unit", "fitted_in", "judges", "factor"}`.

    `who[d]` lists the judges whose noise is the unit (null for a dimension the data fix, or when not
    known); `unit[S, D]` gives the factor's stat on the dimensions it applies to (null when the draws
    were written in the reported unit directly, as `testing.py` does)."""
    from .fitfile import stat

    out = []
    for d, dim in enumerate(dims):
        name = unit_of(dim["scale"]["kind"], dim["id"] in relative)
        noise = name in ("total_noise", "total_pair_noise")
        out.append({"unit": name, "fitted_in": FITTED_IN[name],
                    "judges": list(who[d]) if noise and who is not None and who[d] is not None else None,
                    "factor": stat(np.asarray(unit)[:, d]) if noise and unit is not None else None})
    return out


def old_record(kind: str, relative: bool) -> dict:
    """The `unit` record `read_fit` gives a dimension of a fit of schema 1: the draws are in the engine's
    unit, and who carries it and the factor are not recorded."""
    name = old_unit_of(kind, relative)
    return {"unit": name, "fitted_in": name, "judges": None, "factor": None}
