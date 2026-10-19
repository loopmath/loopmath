"""`recommend --harness-variants [N]` and `--harness PATH|auto`: harness variants priced (`spec/harness.md`).

The search stays factor-free: every candidate is priced as its bare harness label, as before. This block prices
each variant on the configurations its runs used (`RUN_CONFIGS`, most runs first) and on the default pick's and
the goal's, with the variant's factors on every piece, against the bare label on the same 400 joint draws, so
`p_better` and `delta_p_success` are paired. A configuration under whose harness labels none of the variant's
parts has a node is not priced: there the variant would show only the factor scale's spread. `not_priced` says
why, as it does for a run configuration the store no longer holds.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..belief.compose import ranges
from ..belief.design import factor_node

RUN_CONFIGS = 2  # the configurations of a variant's runs it is priced on, most runs first
NOTE = ("Each variant is priced on the configurations its runs used (at most two, most runs first) and on the "
        "default pick's and the goal's, each against the same configuration with the bare harness label. A "
        "configuration under whose harness no part of the variant has data is not priced; `not_priced` says why. "
        "An estimate is the change when a named part is present, pooled over its recorded versions, "
        "not the effect of one edit; a part on every run of a harness cannot be told apart from the harness itself. "
        "The success effects behind p_success, delta_p_success, cost_per_accepted_usd and p_better are not "
        "calibrated: in simulation their 80 and 95% intervals held 60 and 79% of nonzero effects, so these "
        "comparisons lack validated calibration. run_cost_usd rests on the cost effects, whose intervals met their "
        "stated coverage.")
TEXT_NOTE = ("harness variants: {n} priced in --json; their chances, costs per accepted result and p_better rest on "
             "success effects that are not calibrated: spec/harness.md section 9")
TEXT_NONE = "harness variants: none could be priced; not_priced in --json says why"
ROLE_WORDS = {"default_pick": "default pick", "goal": "goal"}
REFIT = ("this fit does not list the configurations the variant's runs used; run `loopmath fit` to price it on "
         "them")


def _r6(d: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def asked_variant(tree: Path, seen: list[dict]) -> dict:
    """The tree `--harness` named, as a variant: its record's digest and factors, and the runs the fit saw."""
    from ..harness import record_from_tree

    rec = record_from_tree(tree)
    same = [v for v in seen if v.get("digest") == rec["digest"]]
    return {"digest": rec["digest"], "source": rec["source"], "factors": rec["factors"],
            "runs": sum(int(v.get("runs") or 0) for v in same),
            "harnesses": sorted({h for v in same for h in v.get("harnesses", [])}),
            "run_configs": [c for v in same for c in v.get("run_configs") or []], "asked": True}


def _labels(cfg: Any) -> list[str]:
    return sorted({s.harness or "unknown" for s in cfg.settings.values()})


def _learned(belief: Any, cfg: Any, factors: list[str]) -> bool:
    """Whether a part of `factors` has a node under one of `cfg`'s harness labels in any fitted head (cost, rate,
    success or a score; the gate head has none)."""
    nodes = {factor_node(h, f) for h in _labels(cfg) for f in factors}
    return any(hs.fitted and not nodes.isdisjoint(hs.index) for hs in belief.heads.values())


def _targets(belief: Any, v: dict, picks: list[tuple[str, Any]], home: Path | None,
             label: Any) -> tuple[list[tuple[Any, list[str], int | None]], list[dict]]:
    """The configurations to price `v` on, each with its roles and its runs, and the ones it is not priced on."""
    from .storeread import config_of_run

    own: list[tuple[Any, list[str], int | None]] = []
    skipped: list[dict] = []
    if "run_configs" not in v and not v.get("asked"):
        skipped.append({"id": None, "role": "runs", "label": None, "reason": REFIT})
    for c in (v.get("run_configs") or [])[:RUN_CONFIGS]:
        cfg = config_of_run(home, str(c["run"]), str(c["id"])) if home is not None else None
        if cfg is None:
            skipped.append({"id": c["id"], "role": "runs", "label": None,
                            "reason": f"the run {c['run']} its configuration is read from is no longer in the store"})
        else:
            own.append((cfg, ["runs"], int(c["runs"])))
    for role, cfg in picks:
        hit = next((t for t in own if t[0].id == cfg.id), None)
        if hit is not None:
            hit[1].append(role)
        else:
            own.append((cfg, [role], None))
    factors = list(v.get("factors") or [])
    priced = []
    for cfg, roles, runs in own:
        if not factors:
            reason = "the variant has no named parts, so it prices as the bare harness label"
        elif not _learned(belief, cfg, factors):
            reason = (f"no part of the variant has data under {', '.join(_labels(cfg))}, the harness this "
                      "configuration runs, so priced there it would show only the prior's spread")
        else:
            priced.append((cfg, roles, runs))
            continue
        skipped += [{"id": cfg.id, "role": role, "label": label(cfg), "reason": reason} for role in roles]
    return priced, skipped


def harness_block(belief: Any, task: Any, rule: Any, rec: Any, *, n: int, asked: dict | None = None,
                  home: Path | None = None) -> dict:
    """The `harness_variants` key: the asked tree first (when given), then the `n` variants the fit read most.
    `home` is the store the variants' run configurations are read from."""
    seen = [dict(v) for v in ((getattr(belief, "design", None) or {}).get("harness_variants") or [])]
    variants = ([asked] if asked else []) + [v for v in seen if not asked or v["digest"] != asked["digest"]][:n]
    picks, ids = [], set()
    for role, cand in (("default_pick", rec.default), ("goal", rec.goal)):
        if cand is not None and cand.config.id not in ids:
            ids.add(cand.config.id)
            picks.append((role, cand.config))
    rescue = float(rec.rescue.usd) if rec.rescue is not None else None
    plans = [(v, *_targets(belief, v, picks, home, rec.label)) for v in variants]
    union = list({cfg.id: cfg for _, priced, _ in plans for cfg, _, _ in priced}.values())
    base = dict(zip((c.id for c in union), belief.predict_with_draws(task, union, rule, rescue_usd=rescue))
                ) if union else {}
    out_variants = []
    for v, targets, skipped in plans:
        cfgs = [cfg for cfg, _, _ in targets]
        got = belief.predict_with_draws(task, cfgs, rule, rescue_usd=rescue,
                                        factors=tuple(v.get("factors") or ())) if cfgs else []
        priced = {}
        cal = {"calibrated": False}  # success-side ranges (`spec/harness.md` section 9)
        for (cfg, roles, runs), (vp, vd) in zip(targets, got):
            bd = base[cfg.id][1]
            priced[cfg.id] = {
                "label": rec.label(cfg), "roles": roles, **({"runs": runs} if runs is not None else {}),
                "p_success": _r6({**ranges(vd["g"], vp.p_success.mean), **cal}),
                "run_cost_usd": _r6({**ranges(vd["run"].sim_usd, vp.cost.usd.mean), "median": vp.cost.usd.median}),
                "cost_per_accepted_usd": _r6({**ranges(vd["ell"], vp.ell.usd.mean), **cal}),
                "p_better": round(float(np.mean(vd["ell"] < bd["ell"])), 6),
                "delta_p_success": _r6({**ranges(vd["g"] - bd["g"]), **cal}),
                "p_better_calibrated": False,
            }
        out_variants.append({**{k: x for k, x in v.items() if k != "run_configs"}, "configs": priced,
                             "not_priced": skipped})
    return {"note": NOTE,
            "configs": [{"id": c.id, "role": role, "label": rec.label(c)} for role, c in picks],
            "variants": out_variants}


def text_lines(hv: dict) -> list[str]:
    """The text output's lines: how many variants are priced, then each of the recommendation's configurations
    some variants are not priced on, with the reason."""
    variants = hv["variants"]
    n = sum(1 for v in variants if v["configs"])
    lines = [TEXT_NOTE.format(n=n) if n else TEXT_NONE]
    skips = Counter((s["role"], s["label"], s["reason"]) for v in variants for s in v["not_priced"]
                    if s["role"] in ROLE_WORDS)
    for (role, label, reason), k in skips.items():
        lines.append(f"  {k} of {len(variants)} not priced on the {ROLE_WORDS[role]} ({label}): {reason}")
    return lines
