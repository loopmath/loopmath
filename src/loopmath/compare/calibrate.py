"""The simulations behind the adopt thresholds in `_thresholds.py` (read by `calibration.py`).

    python -m loopmath.compare.calibrate paths --workers 4 --out DIR      # simulate; writes DIR/paths.npz
    python -m loopmath.compare.calibrate paths --tags check --seed check=5 --workers 4 --out DIR5
    python -m loopmath.compare.calibrate table --out DIR --paths DIR --paths DIR5 --margin 0.8
    python -m loopmath.compare.calibrate stopped --workers 4 --out DIR   # the stopped policy, through paired()

**paths** simulates checks under each null family (the candidate is no better) and under the power
alternatives. A check's P(better) is computed after every pair, from the first up to the largest cap, with no
stop of any kind: the full path, the most chances to adopt by noise. P(better) is exactly what `paired` computes
for those runs (its default seed, its draws); `tests/compare/test_compare_calibration.py` checks that look by
look. Every family is simulated on a tune seed and a check seed; `--tags` and `--seed` simulate a fresh check
seed alone.

**table** sets, for each mode, alpha and cap, the smallest threshold that at most margin * alpha of every
family's tune paths reach by the cap (the largest P(better) seen up to the cap at or above it), and never under
1 - alpha. It checks each threshold on the check seed's paths: the worst family's false adopt rate, with an
exact binomial 95% interval. A bound is stated only where that held-out rate is at or under alpha. The paths
come from one or more `paths` directories, a later one replacing an earlier one's groups. It also writes the power at each cap and the false adopt rates of the round 1 rule (adopt at 0.95 at every look).

**stopped** runs `paired` itself a pair at a time on a third seed, with every stop on (adopt, reject, the
next pair's value, the cap), and counts false adopts under the nulls and adopts under the alternatives; each
adopt must also be a full-path adopt of the same path.
"""

from __future__ import annotations

import argparse
import json
import math
import pprint
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.special import stdtr
from scipy.stats import beta as beta_dist

from . import calibration
from . import model as M

CAPS = (6, 12, 24, 48)
ALPHAS = (0.10, 0.05, 0.025, 0.01)
# Tune to 75% of alpha: room for the Monte Carlo error of the tuned quantile and of the held-out rate, for the
# worst of up to 11 families, and for the coarse P(better) of a few binary pairs. Chosen on check seed 2 (0.9 and
# 0.8 left alpha 0.05 entries over alpha there), so the table is checked on a fresh seed.
MARGIN = 0.75
PATHS = 2000  # paths per family and seed
CHUNK = 50
POWER_PATHS = 1000
POWER_CAPS = (6, 12, 24)
SEEDS = {"tune": 1, "check": 2, "stopped": 3, "power": 4}
TASK_COST_SD = 0.8  # a task's own log-cost level, shared by both of its runs
RUN_COST_SD = 0.5  # a run's log-cost noise
OBJECTIVE = {"cpar": "cpar", "gain": "gain", "score": "gain"}


@dataclass(frozen=True)
class Family:
    """One simulated truth. Binary: each arm's chance of acceptance, task-correlated chances (each task's
    chance from Beta(conc * p, conc * (1 - p)), shared by both arms), the candidate's cost multiple, and
    missingness: unscored and uncosted runs at random, extra base runs with no partner, and mixed models: the
    share of pairs whose two runs (and an extra base run after them) come from a model costing `cheap_cost` times
    the rest, one model per pair. Score: the task sd,
    the run noise (normal or t with 3 df), an optional grid (step, lo, hi, mid, noise sd), the effect, the base
    run noise as a multiple of the candidate's (0: a constant base), and the correlation of the two runs' noise."""

    name: str
    kind: str = "binary"
    pb: float = 0.5
    pc: float = 0.5
    conc: float | None = None
    cost_ratio: float = 1.0
    unscored: float = 0.0
    uncosted: float = 0.0
    extra_base: float = 0.0
    cheap: float = 0.0
    cheap_cost: float = 0.01
    task_sd: float = 0.0
    noise: str = "normal"
    grid: tuple[float, float, float, float, float] | None = None
    effect: float = 0.0
    base_sd: float = 1.0
    rho: float = 0.0


def _bin(p: float, **kw) -> Family:
    return Family(f"binary, both arms accepted at {p:g}", pb=p, pc=p, **kw)


BINARY_NULLS = [_bin(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)] + [
    Family("binary, task-correlated chances, mean 0.3", pb=0.3, pc=0.3, conc=2.0),
    Family("binary, task-correlated chances, mean 0.7", pb=0.7, pc=0.7, conc=2.0),
    Family("binary at 0.5, 10% of runs unscored and 5% uncosted", unscored=0.10, uncosted=0.05),
    Family("binary at 0.5, unequal arms: an extra unpaired base run after 25% of pairs", extra_base=0.25),
    # At 50% to 100% of pairs the gain ran near alpha at a cap of 6 under the thresholds tuned without it.
    Family("binary at 0.5, unequal arms: an extra unpaired base run after 75% of pairs", extra_base=0.75)]
MATCHED_CPAR_NULLS = [
    Family("matched cpar: candidate accepted at 0.6 against 0.4, at 1.5 times the cost", pb=0.4, pc=0.6,
           cost_ratio=1.5),
    Family("matched cpar: candidate accepted at 0.4 against 0.6, at 1/1.5 the cost", pb=0.6, pc=0.4,
           cost_ratio=1 / 1.5)]
SCORE_NULLS = [
    Family("score, normal noise, no task effect", kind="score"),
    Family("score, normal noise, task sd 1", kind="score", task_sd=1.0),
    Family("score, normal noise, task sd 3", kind="score", task_sd=3.0),
    Family("score, heavy-tailed noise (t, 3 df), task sd 1", kind="score", task_sd=1.0, noise="t3"),
    Family("score, judge 0 to 5 on a 0.25 grid", kind="score", task_sd=0.75, grid=(0.25, 0.0, 5.0, 2.5, 0.75)),
    Family("score, judge 1 to 5 in whole points", kind="score", task_sd=1.0, grid=(1.0, 1.0, 5.0, 3.0, 0.7)),
    # A constant base and anti-correlated runs: each ran near alpha under the thresholds tuned without them.
    Family("score, constant base, normal noise on the candidate", kind="score", base_sd=0.0),
    Family("score, judge 0 to 5 on a 0.25 grid, constant base at 2.5", kind="score",
           grid=(0.25, 0.0, 5.0, 2.5, 0.75), base_sd=0.0),
    Family("score, normal noise anti-correlated across the pair (rho -0.5)", kind="score", rho=-0.5)]
# Arms that mix a cheap model and a dear one, as a read and its final share a developer model: cost per run is
# where a lognormal reading of the costs goes wrong.
MIXED_COST_NULLS = [
    Family("matched cpar, mixed models: candidate accepted at 0.6 against 0.4, at 1.5 times the cost, 40% of "
           "pairs at 1/100 the cost", pb=0.4, pc=0.6, cost_ratio=1.5, cheap=0.4),
    Family("binary at 0.5, mixed models: 40% of pairs at 1/100 the cost", cheap=0.4)]
NULLS = {"cpar": BINARY_NULLS + MATCHED_CPAR_NULLS, "gain": BINARY_NULLS, "score": SCORE_NULLS}
ALTERNATIVES = {
    "cpar": [Family("accepted 0.5 to 0.7, same cost", pb=0.5, pc=0.7),
             Family("accepted 0.3 to 0.5, same cost", pb=0.3, pc=0.5),
             Family("accepted 0.7 to 0.9, same cost", pb=0.7, pc=0.9),
             Family("accepted 0.7 both, candidate 30% cheaper", pb=0.7, pc=0.7, cost_ratio=0.7),
             Family("accepted 0.7 both, candidate 50% cheaper", pb=0.7, pc=0.7, cost_ratio=0.5)],
    "gain": [Family("accepted 0.5 to 0.7", pb=0.5, pc=0.7),
             Family("accepted 0.3 to 0.5", pb=0.3, pc=0.5),
             Family("accepted 0.5 to 0.8", pb=0.5, pc=0.8)],
    "score": [Family("normal, task sd 1, effect 0.5 run sd", kind="score", task_sd=1.0, effect=0.5),
              Family("normal, task sd 1, effect 1 run sd", kind="score", task_sd=1.0, effect=1.0),
              Family("judge 0 to 5 on a 0.25 grid, effect 0.5", kind="score", task_sd=0.75,
                     grid=(0.25, 0.0, 5.0, 2.5, 0.75), effect=0.5)]}


# ---------------------------------------------------------------- one simulated check
def _rng(seed: int, mode: str, fam: Family, chunk: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(mode.encode()), zlib.crc32(fam.name.encode()), chunk])


def simulate(fam: Family, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """n pairs of one check. Binary: outcomes 1, 0 or -1 (unscored), log costs (NaN when uncosted), and the
    extra base runs. Score: the two arms' scores."""
    if fam.kind == "score":  # the draws come in round 1's order, so its families' paths are unchanged
        task = fam.task_sd * rng.standard_normal(n)
        if fam.grid is not None:
            step, lo, hi, mid, sd = fam.grid
            zb = rng.standard_normal(n)
            zc = rng.standard_normal(n)
            eb, ec = fam.base_sd * zb, fam.rho * zb + math.sqrt(1 - fam.rho ** 2) * zc
            sb, sc = (np.clip(np.round((mid + task + shift + sd * e) / step) * step, lo, hi)
                      for shift, e in ((0.0, eb), (fam.effect, ec)))
            return {"sb": sb, "sc": sc}
        e = rng.standard_t(3, size=(2, n)) if fam.noise == "t3" else rng.standard_normal((2, n))
        eb, ec = fam.base_sd * e[0], fam.rho * e[0] + math.sqrt(1 - fam.rho ** 2) * e[1]
        return {"sb": task + eb, "sc": task + fam.effect + ec}
    if fam.conc is not None:
        pt = rng.beta(fam.conc * fam.pb, fam.conc * (1 - fam.pb), n)
        pb, pc = pt, pt
    else:
        pb, pc = fam.pb, fam.pc
    ob = (rng.random(n) < pb).astype(np.int8)
    oc = (rng.random(n) < pc).astype(np.int8)
    t = TASK_COST_SD * rng.standard_normal(n)
    xb = t + RUN_COST_SD * rng.standard_normal(n)
    xc = t + math.log(fam.cost_ratio) + RUN_COST_SD * rng.standard_normal(n)
    ob[rng.random(n) < fam.unscored] = -1
    oc[rng.random(n) < fam.unscored] = -1
    xb[rng.random(n) < fam.uncosted] = np.nan
    xc[rng.random(n) < fam.uncosted] = np.nan
    extra = rng.random(n) < fam.extra_base
    eo = (rng.random(n) < fam.pb).astype(np.int8)
    ex = TASK_COST_SD * rng.standard_normal(n) + RUN_COST_SD * rng.standard_normal(n)
    if fam.cheap:  # drawn after the rest, so the other families' paths are unchanged
        lc = np.where(rng.random(n) < fam.cheap, math.log(fam.cheap_cost), 0.0)
        xb, xc, ex = xb + lc, xc + lc, ex + lc
    return {"ob": ob, "oc": oc, "xb": xb, "xc": xc, "extra": extra, "eo": eo, "ex": ex}


def path_p(mode: str, path: dict[str, np.ndarray], looks: int) -> np.ndarray:
    """P(better) after each of the first `looks` pairs, as `paired` computes it for those runs: the data are
    built in paired's order, and the draws come from its default seed. The gain and the score do not read
    costs (the outcome draws are their own stream), so they are left out."""
    floor = calibration.DEFAULT_FLOORS[mode]
    out = np.zeros(looks)
    if mode == "score":
        data = M.Data(kind="score")
        values: list[float] = []
        for k in range(looks):
            b, c = float(path["sb"][k]), float(path["sc"][k])
            data.diffs.append(c - b)
            values += [b, c]
            data.scale = M.score_scale(values)
            nu, loc, scale = M.score_t(data)
            out[k] = float(stdtr(nu, (loc - floor) / scale))
        return out
    costs = mode == "cpar"
    data = M.Data(kind="binary")
    cells = np.zeros(4)
    lone = {"base": [0, 0], "cand": [0, 0]}
    pair_lone = {"base": [], "cand": []}
    run_lone: list[float] = []
    for k in range(looks):
        ob, oc = int(path["ob"][k]), int(path["oc"][k])
        if ob >= 0 and oc >= 0:
            cells[0 if (oc and ob) else 1 if oc else 2 if ob else 3] += 1
        else:
            for arm, o in (("base", ob), ("cand", oc)):
                if o >= 0:
                    lone[arm][0 if o else 1] += 1
        if path["extra"][k]:
            lone["base"][0 if path["eo"][k] else 1] += 1
            if costs:
                run_lone.append(float(path["ex"][k]))
        if costs:
            xb, xc = float(path["xb"][k]), float(path["xc"][k])
            if math.isfinite(xb) and math.isfinite(xc):
                data.cost_pairs.append((xb, xc))
            else:
                for arm, x in (("base", xb), ("cand", xc)):
                    if math.isfinite(x):
                        pair_lone[arm].append(x)
            data.cost_lone = {"base": pair_lone["base"] + run_lone, "cand": list(pair_lone["cand"])}
        data.cells = cells
        data.lone = {a: (v[0], v[1]) for a, v in lone.items()}
        d = M.draws(data, OBJECTIVE[mode], np.random.default_rng(0))
        p = M.p_better(data, d, floor)
        out[k] = 0.0 if p is None else p
    return out


def observations(mode: str, path: dict[str, np.ndarray], k: int):
    """The first k pairs of a path as `compare.Observation`s, for `paired` (and the equality test)."""
    from . import Observation

    base, cand = [], []
    for i in range(k):
        if mode == "score":
            base.append(Observation(f"t{i:03d}", "base", float(path["sb"][i]), 1.0, run=f"b{i:03d}"))
            cand.append(Observation(f"t{i:03d}", "cand", float(path["sc"][i]), 1.0, run=f"c{i:03d}"))
            continue
        for arm, o, x, out in (("base", path["ob"][i], path["xb"][i], base), ("cand", path["oc"][i], path["xc"][i], cand)):
            out.append(Observation(f"t{i:03d}", arm, None if o < 0 else bool(o),
                                   float(math.exp(x)) if math.isfinite(x) else None, run=f"{arm[0]}{i:03d}"))
        if path["extra"][i]:
            base.append(Observation(f"x{i:03d}", "base", bool(path["eo"][i]), float(math.exp(path["ex"][i])),
                                    run=f"x{i:03d}"))
    return base, cand


def rule_for(mode: str) -> str:
    return "score:quality" if mode == "score" else "tests"


# ---------------------------------------------------------------- units of work
def _paths_unit(mode: str, fam: Family, seed: int, chunk: int, n: int, looks: int) -> np.ndarray:
    rng = _rng(seed, mode, fam, chunk)
    return np.stack([path_p(mode, simulate(fam, rng, looks), looks) for _ in range(n)])


def _stopped_unit(mode: str, fam: Family, seed: int, chunk: int, n: int, cap: int,
                  horizon: float = M.HORIZON_RUNS) -> list[dict]:
    """Checks run through `paired` a pair at a time, every stop on. Each row: the verdict, the pairs used,
    whether P(better) matched the full path's at every look, and the full path's largest P(better)."""
    from . import paired

    rng = _rng(seed, mode, fam, chunk)
    rows = []
    for _ in range(n):
        path = simulate(fam, rng, cap)
        full = path_p(mode, path, cap)
        verdict, used, same = "more", cap, True
        for k in range(1, cap + 1):
            base, cand = observations(mode, path, k)
            c = paired(base, cand, rule=rule_for(mode), objective=OBJECTIVE[mode], max_pairs=cap, horizon_runs=horizon)
            same &= abs((c.p_better or 0.0) - full[k - 1]) <= 1e-6
            if c.verdict in ("adopt", "reject"):
                verdict, used = c.verdict, k
                break
        rows.append({"verdict": verdict, "pairs": used, "same_p": bool(same), "full_max": float(full.max()),
                     "threshold": c.adopt_threshold})
    return rows


def _run(units: list[tuple], fn, workers: int, label: str) -> list:
    out: list = [None] * len(units)
    t0 = time.time()
    if workers <= 1:
        for i, u in enumerate(units):
            out[i] = fn(*u)
        return out
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(fn, *u): i for i, u in enumerate(units)}
        for done, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if done % 10 == 0 or done == len(units):
                print(f"{label}: {done}/{len(units)} units, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return out


def _cost(mode: str, fam: Family) -> float:
    lone = fam.unscored > 0 or fam.extra_base > 0
    return {"score": 0.02, "gain": 0.2, "cpar": 1.0}[mode] * (6.0 if lone else 1.0)


# ---------------------------------------------------------------- commands
def cmd_paths(args) -> None:
    looks = max(CAPS)
    seeds = dict(SEEDS, **args.seed)
    units, keys = [], []
    for mode, fams in NULLS.items():
        for fam in fams if mode in args.modes else ():
            if args.families and not any(f in fam.name for f in args.families):
                continue
            for tag in ("tune", "check"):
                for ch in range(args.paths // CHUNK if tag in args.tags else 0):
                    units.append((mode, fam, seeds[tag], ch, CHUNK, looks))
                    keys.append(("null", mode, fam.name, tag))
    for mode, fams in ALTERNATIVES.items():
        for fam in fams if mode in args.modes else ():
            if args.families and not any(f in fam.name for f in args.families):
                continue
            for ch in range(args.power_paths // CHUNK if "power" in args.tags else 0):
                units.append((mode, fam, seeds["power"], ch, CHUNK, max(POWER_CAPS)))
                keys.append(("power", mode, fam.name, "power"))
    order = sorted(range(len(units)), key=lambda i: -_cost(units[i][0], units[i][1]))
    res = _run([units[i] for i in order], _paths_unit, args.workers, "paths")
    grouped: dict[tuple, list[np.ndarray]] = {}
    for i, r in zip(order, res):
        grouped.setdefault(keys[i], []).append((units[i][3], r))
    arrays, index = {}, []
    for j, (key, parts) in enumerate(sorted(grouped.items())):
        arrays[f"a{j}"] = np.concatenate([r for _, r in sorted(parts, key=lambda t: t[0])])
        index.append(list(key))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "paths.npz", **arrays)
    (out / "paths-index.json").write_text(json.dumps({"index": index, "paths": args.paths,
                                                      "power_paths": args.power_paths, "seeds": seeds,
                                                      "tags": sorted(args.tags), "modes": sorted(args.modes)},
                                                     indent=1))
    print(f"paths: wrote {len(index)} groups to {out / 'paths.npz'}", file=sys.stderr)


def interval(k: int, n: int) -> tuple[float, float]:
    """The exact (Clopper-Pearson) binomial 95% interval for k of n."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def _rate(hits: np.ndarray) -> dict:
    k, n = int(hits.sum()), int(hits.size)
    lo, hi = interval(k, n)
    return {"rate": round(k / n, 4), "lo": round(lo, 4), "hi": round(hi, 4), "adopts": k, "sims": n}


def _tuned(maxes: np.ndarray, alpha: float, margin: float = MARGIN) -> float:
    """The smallest threshold at most margin * alpha of these paths reach: just above the order statistic."""
    k = int(math.floor(margin * alpha * maxes.size))
    r = np.sort(maxes)[::-1]
    return 0.0 if k >= r.size else float(r[k]) + 1e-9


def load_paths(dirs: list[str]) -> tuple[dict[tuple, np.ndarray], dict[str, int], dict[str, str]]:
    """The simulated paths of one or more `paths` runs, keyed (kind, mode, family, tag); a later directory's
    group replaces an earlier one's. Also each tag's seed and the directory it came from."""
    data, seed_of, from_dir = {}, {}, {}
    for d in dirs:
        meta = json.loads((Path(d) / "paths-index.json").read_text())
        z = np.load(Path(d) / "paths.npz")
        for j, key in enumerate(meta["index"]):
            data[tuple(key)] = z[f"a{j}"]
            seed_of[tuple(key)], from_dir[tuple(key)] = meta["seeds"][key[3]], str(d)
    seeds, where = {}, {}
    for key, seed in seed_of.items():
        if seeds.setdefault(key[3], seed) != seed:
            raise SystemExit(f"paths: the {key[3]} groups come from more than one seed")
        where.setdefault(key[3], set()).add(from_dir[key])
    return data, seeds, {tag: sorted(v) for tag, v in where.items()}


def cmd_table(args) -> None:
    out = Path(args.out)
    data, seeds, where = load_paths(args.paths or [args.out])
    if seeds["tune"] == seeds["check"]:
        raise SystemExit("table: the tune and check seeds are the same")
    sims = {tag: {data[k].shape[0] for k in data if k[0] == "null" and k[3] == tag} for tag in ("tune", "check")}
    if any(len(v) != 1 for v in sims.values()):
        raise SystemExit(f"table: every family needs the same number of paths per seed, got {sims}")
    sims = {tag: v.pop() for tag, v in sims.items()}
    margin = args.margin
    table, report = {}, {"margin": margin, "alphas": ALPHAS, "caps": CAPS, "looks": calibration.LOOKS,
                         "seeds": seeds, "paths_from": where, "paths": sims["check"], "tune_paths": sims["tune"],
                         "modes": {}}
    for mode, fams in NULLS.items():
        runmax = {f.name: {tag: np.maximum.accumulate(data[("null", mode, f.name, tag)], axis=1)
                           for tag in ("tune", "check")} for f in fams}
        thr: dict[str, dict[str, float]] = {}
        for a in sorted(ALPHAS, reverse=True):
            prev = 0.0
            thr[str(a)] = {}
            for cap in CAPS:
                c = max([_tuned(runmax[f.name]["tune"][:, cap - 1], a, margin) for f in fams] + [1 - a, prev])
                c = math.ceil(c * 1e6) / 1e6
                thr[str(a)][str(cap)] = prev = c
        for i, a in enumerate(sorted(ALPHAS, reverse=True)[1:], 1):  # a smaller alpha is never less strict
            looser = str(sorted(ALPHAS, reverse=True)[i - 1])
            for cap in CAPS:
                thr[str(a)][str(cap)] = max(thr[str(a)][str(cap)], thr[looser][str(cap)])
        held, rows = {}, []
        for a in ALPHAS:
            held[str(a)] = {}
            for cap in CAPS:
                c = thr[str(a)][str(cap)]
                per = {f.name: _rate(runmax[f.name]["check"][:, cap - 1] >= c) for f in fams}
                worst = max(per, key=lambda n: per[n]["rate"])
                held[str(a)][str(cap)] = dict(per[worst], family=worst, verified=per[worst]["rate"] <= a)
                rows.append({"alpha": a, "cap": cap, "threshold": c, "families": per})
        before = {str(cap): {f.name: _rate(runmax[f.name]["check"][:, cap - 1] >= 0.95) for f in fams}
                  for cap in CAPS}
        stops = {}  # adopt at the threshold and reject at 0.05, without the value stop, on the check seed
        for cap in CAPS:
            c = thr["0.05"][str(cap)]
            stops[str(cap)] = {}
            for f in fams:
                p = data[("null", mode, f.name, "check")][:, :cap]
                stops[str(cap)][f.name] = _rate(_first_adopt(p, c, 0.05))
        power = {}
        for alt in ALTERNATIVES[mode]:
            p = data[("power", mode, alt.name, "power")]
            power[alt.name] = {str(cap): {"calibrated": _rate(np.maximum.accumulate(p, axis=1)[:, cap - 1]
                                                              >= thr["0.05"][str(cap)]),
                                          "calibrated_with_reject": _rate(_first_adopt(p[:, :cap], thr["0.05"][str(cap)], 0.05)),
                                          "round1_0.95": _rate(np.maximum.accumulate(p, axis=1)[:, cap - 1] >= 0.95),
                                          "threshold": thr["0.05"][str(cap)]}
                               for cap in POWER_CAPS}
        table[mode] = {"families": [f.name for f in fams], "tune_seed": seeds["tune"], "check_seed": seeds["check"],
                       "sims": sims["check"], "tune_sims": sims["tune"], "margin": margin, "thresholds": thr,
                       "held_out": held}
        report["modes"][mode] = {"families": [asdict(f) for f in fams], "thresholds": thr, "held_out": held,
                                 "per_family": rows, "round1_rule": before, "with_reject_stop": stops,
                                 "power": power, "alternatives": [asdict(f) for f in ALTERNATIVES[mode]]}
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    target = Path(args.table) if args.table else Path(__file__).with_name("_thresholds.py")
    target.write_text('"""Adopt thresholds and their held-out false adopt rates. Generated by\n'
                      '`python -m loopmath.compare.calibrate table`; do not edit by hand."""\n\n'
                      f"TABLE = {pprint.pformat(table, width=110, sort_dicts=True)}\n")
    print(f"table: wrote {target} and {out / 'calibration.json'}", file=sys.stderr)


def _first_adopt(p: np.ndarray, adopt: float, reject: float) -> np.ndarray:
    """Per path: does P(better) reach `adopt` before it falls to `reject`."""
    hit_a = p >= adopt
    hit_r = p <= reject
    n = p.shape[1]
    first_a = np.where(hit_a.any(axis=1), hit_a.argmax(axis=1), n)
    first_r = np.where(hit_r.any(axis=1), hit_r.argmax(axis=1), n)
    return first_a < first_r


def cmd_stopped(args) -> None:
    units, keys = [], []
    for kind, table, caps, horizons in (("null", NULLS, args.caps, [M.HORIZON_RUNS]),
                                        ("power", ALTERNATIVES, args.power_caps, args.power_horizons)):
        for mode, fams in table.items() if kind in args.kinds else ():
            for fam in fams:
                lone = fam.unscored > 0 or fam.extra_base > 0
                n = args.power_paths if kind == "power" else (args.lone_paths if lone else args.paths)
                for cap in caps:
                    seed = SEEDS["stopped"] + cap + (100 if kind == "power" else 0)
                    for h in horizons if mode != "score" else horizons[:1]:  # a score has no dollar value
                        for ch in range(max(n // args.chunk, 1)):
                            units.append((mode, fam, seed, ch, args.chunk, cap, h))
                            keys.append((kind, mode, fam.name, cap, h))
    order = sorted(range(len(units)), key=lambda i: -_cost(units[i][0], units[i][1]) * units[i][5])
    res = _run([units[i] for i in order], _stopped_unit, args.workers, "stopped")
    grouped: dict[tuple, list] = {}
    for i, r in zip(order, res):
        grouped.setdefault(keys[i], []).extend(r)
    out = Path(args.out)
    path = out / "stopped.json"  # a run of one kind keeps the other kind's results
    report = json.loads(path.read_text()) if path.is_file() else {"null": {}, "power": {}}
    report |= {"seed": SEEDS["stopped"], "looks": calibration.LOOKS, "horizon_runs": M.HORIZON_RUNS,
               "look_sims": M.LOOK_SIMS,
               "thresholds_from": calibration.table_in_use().get("cpar", {}).get("check_seed")}
    report |= {kind: {} for kind in args.kinds}
    for (kind, mode, name, cap, h), rows in sorted(grouped.items()):
        thr = rows[0]["threshold"]
        adopt = np.array([r["verdict"] == "adopt" for r in rows])
        full = np.array([r["full_max"] >= thr for r in rows])
        at = report[kind].setdefault(mode, {}).setdefault(str(cap), {})
        at = at.setdefault(f"horizon {h:g}", {}) if kind == "power" else at
        at[name] = dict(
            _rate(adopt), threshold=thr, reject=sum(r["verdict"] == "reject" for r in rows),
            more=sum(r["verdict"] == "more" for r in rows),
            mean_pairs=round(float(np.mean([r["pairs"] for r in rows])), 2),
            full_path=_rate(full), same_p=all(r["same_p"] for r in rows),
            adopts_outside_full=int(np.sum(adopt & ~full)))
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"stopped: wrote {path}", file=sys.stderr)


def _seed(value: str) -> tuple[str, int]:
    tag, _, n = value.partition("=")
    if tag not in SEEDS or not n.isdigit():
        raise argparse.ArgumentTypeError(f"{value}: TAG=N with TAG one of {', '.join(SEEDS)}")
    return tag, int(n)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m loopmath.compare.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--paths", type=int, default=PATHS)
    p.add_argument("--power-paths", type=int, default=POWER_PATHS)
    p.add_argument("--tags", type=lambda v: set(v.split(",")), default={"tune", "check", "power"},
                   help="which of tune, check and power to simulate")
    p.add_argument("--modes", type=lambda v: set(v.split(",")), default=set(NULLS), help="cpar, gain, score")
    p.add_argument("--families", action="append", help="only the families whose name contains this (repeatable)")
    p.add_argument("--seed", type=_seed, action="append", default=[], help="TAG=N: another seed for a tag")
    p.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--paths", action="append", help="a paths directory (repeatable; default --out)")
    t.add_argument("--margin", type=float, default=MARGIN)
    t.add_argument("--table", help="where to write the thresholds module (default: the package's _thresholds.py)")
    s = sub.add_parser("stopped")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--paths", type=int, default=200)
    s.add_argument("--lone-paths", type=int, default=100, help="paths for the families with lone runs (slower)")
    s.add_argument("--power-paths", type=int, default=100)
    s.add_argument("--chunk", type=int, default=20)
    s.add_argument("--caps", type=lambda v: [int(x) for x in v.split(",")], default=[12, 24])
    s.add_argument("--power-caps", type=lambda v: [int(x) for x in v.split(",")], default=[24])
    s.add_argument("--power-horizons", type=lambda v: [float(x) for x in v.split(",")], default=[M.HORIZON_RUNS])
    s.add_argument("--kinds", type=lambda v: set(v.split(",")), default={"null", "power"},
                   help="null, power or both")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "paths":
        args.seed = dict(args.seed)
    {"paths": cmd_paths, "table": cmd_table, "stopped": cmd_stopped}[args.cmd](args)


if __name__ == "__main__":
    main()
