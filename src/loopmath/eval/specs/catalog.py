"""The catalog specs of `eval simulate`, one builder per spec.

Every spec shares a small cast, so the studies read alike:

- arms built from three made-up step settings: `model-small` (family `beta`), `model-large` (family
  `alpha`) and `model-mid` (family `gamma`);
- a person, `reviewer`, who anchors the scale and scores a share of the targets;
- model judges at two grades, `strong` (small bias and noise) and `cheap` (lenient and noisy).

True values are on the fitting scale: `linear`, `fraction` and `levels` dimensions on [0, 1], `binary`
and `items` dimensions in probit units. On a `binary` dimension the reviewer's call noise is the unit the
spec is written in, so its `sigma` is 1; `truth.json` and the fit report the dimension in the reviewer's
total noise of one reading, `sqrt(sigma^2 + tau^2)` (1.118 at the reviewer's tau of 0.5), so every number
there is the spec's divided by that. Arm means on `linear` and `fraction` dimensions stay near the middle
of the scale, so that almost no reading falls outside it and has to be cut to the range (`truth.json`
counts them).

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
    "unit": {"reviewer": (0.0, 0.03, 0.04), "strong": (0.03, 0.05, 0.04), "cheap": (0.08, 0.08, 0.07)},
    "levels": {"reviewer": (0.0, 0.06, 0.08), "strong": (0.04, 0.08, 0.08), "cheap": (0.12, 0.15, 0.15)},
    "binary": {"reviewer": (0.0, 0.5, 1.0), "strong": (0.2, 0.8, 0.8), "cheap": (0.8, 1.2, 1.5)},
    "items": {"reviewer": (0.0, 0.15, 0.20), "strong": (0.10, 0.20, 0.20), "cheap": (0.30, 0.35, 0.40)},
}
# Per scale group: (sd_input, sd_run).
SPREAD = {"unit": (0.07, 0.05), "levels": (0.18, 0.15), "binary": (1.2, 1.0), "items": (0.4, 0.3)}


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
    return spec(comps, two_arms({"depth": 0.42, "clarity": 0.46}, {"depth": 0.54, "clarity": 0.52}), cast(comps),
                usual_design(), corr_input=0.4, corr_run=0.3)


def build_one_component() -> dict:
    comps = [linear("quality", "Quality of the answer")]
    return spec(comps, two_arms({"quality": 0.42}, {"quality": 0.54}), cast(comps), usual_design())


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
    return spec(comps, two_arms({"depth": 0.42, "numbers_right": 0.8}, {"depth": 0.54, "numbers_right": 2.0}),
                cast(comps), usual_design(repeats=3, repeat_share=0.5))


# ---------------------------------------------------------------------------------------------------
# Check items, and every scale kind in one rubric.

def _case_items(item_sd: float) -> dict:
    comps = [items("coverage", "Coverage of the case points"), items("caveats", "Caveats that apply")]
    design = usual_design(items=[
        {"component": "coverage", "per_input": [3, 6], "as": "items", "item_sd": item_sd},
        {"component": "caveats", "per_input": [4, 8], "missing_share": 0.1, "as": "count", "item_sd": item_sd}])
    return spec(comps, two_arms({"coverage": 0.2, "caveats": 0.0}, {"coverage": 0.8, "caveats": 0.5}), cast(comps),
                design)


def build_case_items() -> dict:
    return _case_items(0.0)


def build_case_items_uneven() -> dict:
    # Items of one case differ in difficulty, so k of n is no longer binomial: a stress test of that reading.
    return _case_items(0.8)


def build_mixed_rubric() -> dict:
    comps = [binary("numbers_right", "The numbers are right"), levels("clarity", "Clarity", ["poor", "ok", "good"]),
             linear("depth", "Depth of the case"),
             {"id": "filler", "title": "Share of sentences that are filler", "scale": {"kind": "fraction"},
              "better": "lower"},
             items("coverage", "Coverage of the case points")]
    # Means are on the fitting scale, where higher is better: 0.45 on `filler` is a filler share of 0.55.
    low = {"numbers_right": 0.8, "clarity": 0.40, "depth": 0.42, "filler": 0.45, "coverage": 0.2}
    high = {"numbers_right": 2.0, "clarity": 0.62, "depth": 0.54, "filler": 0.53, "coverage": 0.8}
    return spec(comps, two_arms(low, high), cast(comps),
                usual_design(repeat_share=0.5, items=[{"component": "coverage", "per_input": [3, 6]}]))


# ---------------------------------------------------------------------------------------------------
# Value: gates, an overall judgement, pairs.

def build_gated() -> dict:
    # Two gates on correlated components: the chance to pass both is well above the product of the two.
    comps = [binary("numbers_right", "The numbers are right"), levels("clarity", "Clarity", ["poor", "ok", "good"]),
             linear("depth", "Depth of the case")]
    arms = [arm("baseline", {"final": SMALL}, {"numbers_right": 0.6, "clarity": 0.40, "depth": 0.42}, 0.12),
            arm("mid", {"final": MID}, {"numbers_right": 1.0, "clarity": 0.50, "depth": 0.48}, 0.40),
            arm("strong", {"final": LARGE}, {"numbers_right": 1.6, "clarity": 0.62, "depth": 0.54}, 0.85)]
    corr = {"default": 0.2, "pairs": [["numbers_right", "clarity", 0.7]]}
    value = {"weights": {"clarity": 1, "depth": 2},
             "gates": [{"component": "numbers_right"}, {"component": "clarity", "at_least": "ok"}],
             "gated_value": 0, "objective": {"kind": "best_value", "reference": "baseline"}}
    return spec(comps, arms, cast(comps), usual_design(repeat_share=0.5), corr_input=corr, corr_run=corr, value=value)


def build_pairs() -> dict:
    # `depth` has scores and pairs. `overall` has pairs only: it is relative, in units of the reviewer's
    # pair noise, and its arm means sum to 0. Each judge compares on one dimension, so its tie width and
    # position bias are in one unit.
    depth = [linear("depth", "Depth of the case")]
    comps = depth + [linear("overall", "Overall", role="overall")]
    arms = two_arms({"depth": 0.42, "overall": -0.4}, {"depth": 0.54, "overall": 0.4})
    people = reviewer(depth, position=0.2, tie_width=0.5)
    people["truth"]["tau"]["overall"], people["truth"]["sigma_pair"] = 0.4, {"overall": 1.0}
    strong = model_judge("strong", LARGE, noise(depth, "strong", position=0.03, tie_width=0.04,
                                                sigma_pair={"depth": 0.05}), 0.04)
    cheap = model_judge("cheap", SMALL, noise(depth, "cheap", position=0.5, tie_width=0.3), 0.004)
    cheap["truth"]["tau"]["overall"], cheap["truth"]["sigma_pair"] = 0.7, {"overall": 1.3}
    design = {"scores": [{"judge": "strong", "repeats": 2, "components": ["depth"]},
                         {"judge": "cheap", "repeats": 2, "components": ["depth"]}],
              "human": [{"judge": "reviewer", "rate": 0.5, "repeat_share": 0.25, "components": ["depth"]}],
              "pairs": [{"judge": "reviewer", "component": "overall", "orders": "both"},
                        {"judge": "cheap", "component": "overall", "orders": "both", "repeats": 2},
                        {"judge": "strong", "component": "depth", "orders": "both", "repeats": 2}]}
    out = spec(comps, arms, [people, strong, cheap], design, corr_input=0.5, corr_run=0.5)
    out["quality"]["sd_input"]["overall"], out["quality"]["sd_run"]["overall"] = 0.6, 0.8
    return out


def _mix(a: float, b: float, s1: float, s2: float, r: float, share: float) -> tuple[float, float, float]:
    """An overall quality `a x q1 + b x q2 + rest`, where the criteria explain `share` of its variance.

    Returns its sd and its correlation with each criterion, given the criteria's sds and correlation."""
    explained = a * a * s1 * s1 + b * b * s2 * s2 + 2 * a * b * r * s1 * s2
    sd = math.sqrt(explained / share)
    return sd, (a * s1 * s1 + b * r * s1 * s2) / (s1 * sd), (b * s2 * s2 + a * r * s1 * s2) / (s2 * sd)


def build_overall_vs_criteria() -> dict:
    # overall = 0.1 + 0.5 depth + 0.3 clarity + a part no criterion measures. The criteria explain 60% of
    # its variance at the input and at the run level. `polished` leads on the weighted criteria and
    # trails `strong` on the overall quality, so the two rankings differ.
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity"),
             linear("overall", "Overall", role="overall")]
    a, b, share, r = 0.5, 0.3, 0.6, 0.3

    def overall(depth: float, clarity: float, rest: float) -> dict:
        return {"depth": depth, "clarity": clarity, "overall": round(0.1 + a * depth + b * clarity + rest, 6)}

    arms = [arm("baseline", {"final": SMALL}, overall(0.42, 0.46, 0.0), 0.12),
            arm("strong", {"final": LARGE}, overall(0.54, 0.50, 0.04), 0.85),
            arm("polished", {"final": MID}, overall(0.50, 0.60, -0.06), 0.40)]
    out = spec(comps, arms, cast(comps), usual_design(),
               value={"weights": {"depth": 5, "clarity": 3}, "objective": {"kind": "best_value"}})
    for level in ("input", "run"):
        sds = out["quality"][f"sd_{level}"]
        sds["overall"], with_depth, with_clarity = _mix(a, b, sds["depth"], sds["clarity"], r, share)
        out["quality"][f"corr_{level}"] = {"pairs": [["depth", "clarity", r], ["depth", "overall", with_depth],
                                                     ["clarity", "overall", with_clarity]]}
    return out


# ---------------------------------------------------------------------------------------------------
# Judges: family effects, a shown reference, halo, panel calls.

def _three_families() -> list[dict]:
    return [arm("baseline", {"final": SMALL}, {"depth": 0.42, "clarity": 0.46}, 0.12),
            arm("mid", {"final": MID}, {"depth": 0.48, "clarity": 0.49}, 0.40),
            arm("strong", {"final": LARGE}, {"depth": 0.54, "clarity": 0.52}, 0.85)]


def build_own_family() -> dict:
    # `strong` and `cheap` each favour outputs written by their own model family. `mid` does not.
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity")]
    judges = [reviewer(comps),
              model_judge("strong", LARGE, noise(comps, "strong", family={"alpha": 0.06}), 0.04),
              model_judge("cheap", SMALL, noise(comps, "cheap", family={"beta": 0.08}), 0.004),
              model_judge("mid", MID, noise(comps, "strong", scale=1.3), 0.02)]
    design = usual_design()
    design["scores"].append({"judge": "mid", "repeats": 2})
    return spec(comps, _three_families(), judges, design)


def build_reference_family() -> dict:
    # Two judges are run with and without a reference answer written by a model of family `gamma`.
    # With the reference each favours outputs from that family. The reference also changes the judge's
    # noise: it helps the cheap judge more than the strong one.
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity")]
    shown = {"shown": True, "author": {"kind": "model", "model": dict(MID["model"])}, "version": "r1"}
    judges = [reviewer(comps),
              model_judge("strong", LARGE, noise(comps, "strong"), 0.04),
              model_judge("strong-ref", LARGE, noise(comps, "strong", scale=0.9, family={"gamma": 0.07}), 0.05,
                          reference=shown),
              model_judge("cheap", SMALL, noise(comps, "cheap"), 0.004),
              model_judge("cheap-ref", SMALL, noise(comps, "cheap", scale=0.7, family={"gamma": 0.10}), 0.006,
                          reference=shown)]
    design = usual_design()
    design["scores"] += [{"judge": "strong-ref", "repeats": 2}, {"judge": "cheap-ref", "repeats": 2}]
    return spec(comps, _three_families(), judges, design)


def build_halo() -> dict:
    # The cheap judge scores all three criteria in one call, with a strong halo at both levels.
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity"), linear("structure", "Structure")]
    judges = [reviewer(comps, halo_target=0.1),
              model_judge("strong", LARGE, noise(comps, "strong", halo_target=0.2), 0.04),
              model_judge("cheap", SMALL, noise(comps, "cheap", halo_target=0.6, halo_call=0.4), 0.004, mode="panel")]
    low = {"depth": 0.42, "clarity": 0.46, "structure": 0.44}
    high = {"depth": 0.54, "clarity": 0.52, "structure": 0.52}
    design = usual_design()
    design["scores"][1]["repeats"] = 3
    return spec(comps, two_arms(low, high), judges, design, corr_input=0.2, corr_run=0.2)


def build_panel_vs_per_component() -> dict:
    # One model, effort and prompt, run in both modes. The panel calls cost a third as many calls and
    # lose accuracy: more bias, more noise, and a halo inside the call.
    comps = [linear("depth", "Depth of the case"), linear("clarity", "Clarity"),
             binary("numbers_right", "The numbers are right")]
    panel = noise(comps, "strong", scale=1.5, halo_target=0.3, halo_call=0.5)
    panel["bias"] = {"depth": 0.06, "clarity": 0.06, "numbers_right": 0.5}
    judges = [reviewer(comps),
              model_judge("strong", LARGE, noise(comps, "strong", halo_target=0.2), 0.04),
              model_judge("strong-panel", LARGE, panel, 0.06, mode="panel")]
    low = {"depth": 0.42, "clarity": 0.46, "numbers_right": 0.8}
    high = {"depth": 0.54, "clarity": 0.52, "numbers_right": 2.0}
    design = {"scores": [{"judge": "strong", "repeats": 2}, {"judge": "strong-panel", "repeats": 2}],
              "human": [{"judge": "reviewer", "rate": 0.5, "repeat_share": 0.5}]}
    return spec(comps, two_arms(low, high), judges, design)


# ---------------------------------------------------------------------------------------------------
# Steps.

def build_steps_swap() -> dict:
    # Three steps. Swapping the model at `research` lifts the final depth. Swapping it at `draft` lifts
    # the draft's own quality and nothing at the final step, although draft structure and final clarity
    # move together run by run (correlation 0.6). `all-strong` falls 0.02 short of additive on depth.
    comps = [linear("sources", "Quality of the sources", applies_to=["research"]),
             linear("structure", "Structure of the draft", applies_to=["draft"]),
             linear("depth", "Depth of the case"), linear("clarity", "Clarity")]

    def setting(research: dict, draft: dict, final: dict) -> dict:
        return {"research": research, "draft": draft, "final": final}

    def mean(sources: float, structure: float, depth: float, clarity: float) -> dict:
        return {"sources": sources, "structure": structure, "depth": depth, "clarity": clarity}

    arms = [arm("baseline", setting(SMALL, SMALL, SMALL), mean(0.42, 0.42, 0.42, 0.46), 0.30),
            arm("research-strong", setting(LARGE, SMALL, SMALL), mean(0.57, 0.42, 0.52, 0.47), 0.50),
            arm("draft-strong", setting(SMALL, LARGE, SMALL), mean(0.42, 0.57, 0.42, 0.46), 0.55),
            arm("final-strong", setting(SMALL, SMALL, LARGE), mean(0.42, 0.42, 0.48, 0.52), 0.60),
            arm("all-strong", setting(LARGE, LARGE, LARGE), mean(0.57, 0.57, 0.56, 0.53), 1.15)]
    corr_run = {"default": 0.1, "pairs": [["structure", "clarity", 0.6], ["sources", "depth", 0.4],
                                          ["depth", "clarity", 0.3]]}
    value = {"weights": {"depth": 2, "clarity": 1}, "objective": {"kind": "best_value", "reference": "baseline"}}
    return spec(comps, arms, cast(comps), usual_design(), corr_input=0.3, corr_run=corr_run,
                steps=["research", "draft", "final"], value=value)


CATALOG: dict[str, tuple[str, Callable[[], dict]]] = {
    "linear": ("two linear criteria, two arms, a reviewer and two model judges with repeats", build_linear),
    "one-component": ("one linear criterion: the three-term split of judge error", build_one_component),
    "binary-checks": ("three yes or no checks, with a lenient cheap judge", build_binary_checks),
    "levels-3": ("two criteria on a 3-level scale", build_levels_3),
    "levels-5": ("two criteria on a 5-level scale, one arm near the top", build_levels_5),
    "cheap-vs-strong-judge": ("a linear and a binary criterion, three repeats per judge, judge costs",
                              build_cheap_vs_strong_judge),
    "case-items": ("check items written per case, read item by item and as a count", build_case_items),
    "mixed-rubric": ("every scale kind in one rubric, one of them lower-is-better", build_mixed_rubric),
    "gated": ("two gates on correlated components, with a value file", build_gated),
    "pairs": ("scores and pairs on one criterion, pairs only on the overall judgement", build_pairs),
    "overall-vs-criteria": ("an overall judgement that depends on a part no criterion measures",
                            build_overall_vs_criteria),
    "own-family": ("two judges that favour outputs from their own model family", build_own_family),
    "reference-family": ("judges run with and without a reference answer from a third family",
                         build_reference_family),
    "halo": ("three criteria and a cheap judge with a strong halo, in panel calls", build_halo),
    "panel-vs-per-component": ("one judge in panel and per-component mode: the batching loss",
                               build_panel_vs_per_component),
    "steps-swap": ("three steps, single-step swaps, and a step that correlates without driving", build_steps_swap),
    "case-items-uneven": ("case-items with items of unequal difficulty: a stress test of k of n",
                          build_case_items_uneven),
}
