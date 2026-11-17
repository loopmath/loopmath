"""The catalog specs of `eval simulate`, one builder per spec.

Every spec shares a small cast, so the studies read alike:

- arms built from three made-up step settings: `model-small` (family `beta`), `model-large` (family
  `alpha`) and `model-mid` (family `gamma`);
- a person, `reviewer`, who anchors the scale and scores a share of the targets;
- model judges at two grades, `strong` (small bias and noise) and `cheap` (lenient and noisy).

True values are on the fitting scale: `linear`, `fraction` and `levels` dimensions on [0, 1], `binary`
and `items` dimensions in probit units. On a `binary` dimension the reviewer's call noise is the unit,
so its `sigma` is 1.

Every number here is made up.
"""

from __future__ import annotations

import math
from typing import Callable

SCHEMA = "loopmath.eval-sim/1"
INPUTS = 40
SMALL = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
LARGE = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}
MID = {"model": {"id": "model-mid", "family": "gamma"}, "effort": "medium"}

# Per scale group and judge grade: (bias, tau, sigma).
NOISE = {
    "unit": {"reviewer": (0.0, 0.03, 0.04), "strong": (0.03, 0.05, 0.04), "cheap": (0.10, 0.10, 0.09)},
    "levels": {"reviewer": (0.0, 0.06, 0.08), "strong": (0.04, 0.08, 0.08), "cheap": (0.12, 0.15, 0.15)},
    "binary": {"reviewer": (0.0, 0.5, 1.0), "strong": (0.2, 0.8, 0.8), "cheap": (0.8, 1.2, 1.5)},
    "items": {"reviewer": (0.0, 0.15, 0.20), "strong": (0.10, 0.20, 0.20), "cheap": (0.30, 0.35, 0.40)},
}
# Per scale group: (sd_input, sd_run).
SPREAD = {"unit": (0.08, 0.06), "levels": (0.18, 0.15), "binary": (1.2, 1.0), "items": (0.4, 0.3)}


def _group(component: dict) -> str:
    kind = component["scale"]["kind"]
    return "unit" if kind in ("linear", "fraction") else kind


def linear(cid: str, title: str, **more) -> dict:
    return {"id": cid, "title": title, "scale": {"kind": "linear", "range": [0, 10]}, **more}


def binary(cid: str, title: str, **more) -> dict:
    return {"id": cid, "title": title, "scale": {"kind": "binary"}, **more}


def levels(cid: str, title: str, names: list, **more) -> dict:
    return {"id": cid, "title": title, "scale": {"kind": "levels", "levels": list(names)}, **more}


def items(cid: str, title: str, **more) -> dict:
    return {"id": cid, "title": title, "scale": {"kind": "items"}, **more}


def noise(components: list[dict], grade: str, scale: float = 1.0, **more) -> dict:
    """A judge's truth from the presets: `bias`, `tau` and `sigma` per component, `scale` times the noise."""
    truth: dict = {"bias": {}, "tau": {}, "sigma": {}}
    for comp in components:
        bias, tau, sigma = NOISE[_group(comp)][grade]
        truth["bias"][comp["id"]], truth["tau"][comp["id"]] = bias, tau * scale
        truth["sigma"][comp["id"]] = sigma * scale
    truth.update(more)
    return truth


def spread(components: list[dict]) -> dict:
    """`sd_input` and `sd_run` per component from the presets."""
    return {"sd_input": {comp["id"]: SPREAD[_group(comp)][0] for comp in components},
            "sd_run": {comp["id"]: SPREAD[_group(comp)][1] for comp in components}}


def reviewer(components: list[dict], **more) -> dict:
    return {"id": "reviewer", "kind": "person", "blinded": True, "truth": noise(components, "reviewer", **more)}


def model_judge(jid: str, setting: dict, truth: dict, cost: float, **more) -> dict:
    return {"id": jid, "kind": "model", "model": dict(setting["model"]), "effort": setting["effort"], "prompt": "v1",
            "blinded": True, "cost_usd_per_call": cost, "truth": truth, **more}


def cast(components: list[dict]) -> list[dict]:
    """The usual three judges: the reviewer, `strong` and `cheap`."""
    return [reviewer(components), model_judge("strong", LARGE, noise(components, "strong"), 0.04),
            model_judge("cheap", SMALL, noise(components, "cheap"), 0.004)]


def arm(aid: str, settings: dict, mean: dict, cost: float) -> dict:
    return {"id": aid, "steps": [{"step": step, **setting} for step, setting in settings.items()], "mean": mean,
            "cost_usd": cost}


def two_arms(low: dict, high: dict) -> list[dict]:
    """`baseline` on the small model and `strong` on the large one, with the given mean vectors."""
    return [arm("baseline", {"final": SMALL}, low, 0.12), arm("strong", {"final": LARGE}, high, 0.85)]


def usual_design(repeats: int = 2, rate: float = 0.5, repeat_share: float = 0.25, **more) -> dict:
    """`strong` and `cheap` score every target `repeats` times; the reviewer scores a uniform share."""
    return {"scores": [{"judge": "strong", "repeats": repeats}, {"judge": "cheap", "repeats": repeats}],
            "human": [{"judge": "reviewer", "rate": rate, "selection": "uniform", "repeat_share": repeat_share}],
            **more}


def spec(components: list[dict], arms: list[dict], judges: list[dict], design: dict, *, corr_input=0.3,
         corr_run=0.3, steps: list[str] | None = None, quality: dict | None = None, value: dict | None = None,
         rubric_id: str = "brief") -> dict:
    rubric: dict = {"id": rubric_id, "version": "1", "components": components}
    if steps is not None:
        rubric["steps"] = steps
    out = {"schema": SCHEMA, "name": "", "title": "", "inputs": INPUTS, "runs_per_input": 1, "arms": arms,
           "rubric": rubric, "quality": {**spread(components), "corr_input": corr_input, "corr_run": corr_run,
                                         **(quality or {})},
           "judges": {"anchor": ["reviewer"], "min_anchor_targets": 6, "judges": judges}, "design": design}
    if value is not None:
        out["value"] = value
    return out


# ---------------------------------------------------------------------------------------------------
# The basic specs: one scale kind each.

def build_linear() -> dict:
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity")]
    return spec(comps, two_arms({"depth": 0.50, "clarity": 0.55}, {"depth": 0.64, "clarity": 0.60}), cast(comps),
                usual_design(), corr_input=0.4, corr_run=0.3)


def build_one_component() -> dict:
    comps = [linear("quality", "Quality of the answer")]
    return spec(comps, two_arms({"quality": 0.50}, {"quality": 0.62}), cast(comps), usual_design())


def build_binary_checks() -> dict:
    comps = [binary("numbers_right", "The numbers are right"), binary("cites_sources", "Every claim has a source"),
             binary("follows_format", "The answer follows the format")]
    low = {"numbers_right": 0.8, "cites_sources": 0.4, "follows_format": 1.6}
    high = {"numbers_right": 2.0, "cites_sources": 1.2, "follows_format": 1.8}
    return spec(comps, two_arms(low, high), cast(comps), usual_design(repeat_share=0.5), corr_input=0.3, corr_run=0.2)


def build_levels_3() -> dict:
    names = ["poor", "ok", "good"]
    comps = [levels("clarity", "Clarity", names), levels("structure", "Structure", names)]
    return spec(comps, two_arms({"clarity": 0.40, "structure": 0.45}, {"clarity": 0.62, "structure": 0.55}),
                cast(comps), usual_design())


def build_levels_5() -> dict:
    # The strong arm sits near the top of the scale, where reading levels as plain numbers squeezes differences.
    comps = [levels("clarity", "Clarity", [1, 2, 3, 4, 5]), levels("reasoning", "Reasoning", [1, 2, 3, 4, 5])]
    return spec(comps, two_arms({"clarity": 0.62, "reasoning": 0.50}, {"clarity": 0.90, "reasoning": 0.70}),
                cast(comps), usual_design())


def build_cheap_vs_strong_judge() -> dict:
    # Three repeats per judge, so each judge's run-to-run noise and its item-specific disagreement separate well.
    comps = [linear("depth", "Depth of the case"), binary("numbers_right", "The numbers are right")]
    return spec(comps, two_arms({"depth": 0.50, "numbers_right": 0.8}, {"depth": 0.64, "numbers_right": 2.0}),
                cast(comps), usual_design(repeats=3, repeat_share=0.5))


CATALOG: dict[str, tuple[str, Callable[[], dict]]] = {
    "linear": ("two linear criteria, two arms, a reviewer and two model judges with repeats", build_linear),
    "one-component": ("one linear criterion: the three-term split of judge error", build_one_component),
    "binary-checks": ("three yes or no checks, with a lenient cheap judge", build_binary_checks),
    "levels-3": ("two criteria on a 3-level scale", build_levels_3),
    "levels-5": ("two criteria on a 5-level scale, one arm near the top", build_levels_5),
    "cheap-vs-strong-judge": ("a linear and a binary criterion, three repeats per judge, judge costs",
                              build_cheap_vs_strong_judge),
}
