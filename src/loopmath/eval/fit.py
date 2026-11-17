"""`eval fit`: runs `model.py`, then the engine, then writes the fit file.

    loopmath eval fit STUDY --rubric rubric.json --judges judges.json --seed 0 --out STUDY/fit.json

1. `model.build` reads the study and prints the identification report. With `--check-only` it stops there.
2. The engine samples (`engine_pymc.sample`). Without the [bayes] extra the command prints the install
   line, returns 1 and writes nothing.
3. The draws are thinned to `--keep-draws`, a parameter that does not exist is set to NaN (each NaN has
   a term in state `absent` with its reason), and `fitfile.assemble` and `fitfile.write_fit` write
   `fit.json` and its draws.

Convergence. `diagnostics.checks` lists three checks over the parameters that exist: split R-hat at most
`RHAT_MAX`, effective sample size at least `ESS_MIN`, and divergent transitions at most
`DIVERGENT_SHARE` of the draws. `converged` is false when any fails; the fit is written either way.

Defaults: 4 chains of 1000 draws after 1000 warm-up draws. `fit_design` is the same path as a function,
for tests and other callers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..output import EXIT_OK, EXIT_USER, emit_json
from . import contracts as C
from . import fitfile as F
from . import model as M

OUTPUT_SCHEMA = "loopmath.eval-fit-report/1"   # what `eval fit --json` prints; the fit file is loopmath.eval-fit/1
RHAT_MAX = 1.05
ESS_MIN = 100.0
DIVERGENT_SHARE = 0.01
DEFAULT_DRAWS, DEFAULT_CHAINS = 1000, 4
CHECKED = ("mu", "sd_input", "sd_run", "bias", "tau", "sigma", "halo_target", "halo_call", "family",
           "sd_judge_input", "sigma_pair", "position", "tie_width")


def split_rhat_ess(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split R-hat and effective sample size of `x[chains, draws, P]`, per column.

    R-hat compares the two halves of every chain. The effective sample size sums the autocorrelation in
    pairs until a pair turns negative (Geyer's rule), with the pair sums kept from rising."""
    chains, draws, _ = x.shape
    half = draws // 2
    parts = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    within = parts.var(axis=1, ddof=1).mean(axis=0)
    between = parts.mean(axis=1).var(axis=0, ddof=1)
    safe = np.where(within > 0, within, 1.0)
    rhat = np.sqrt(((half - 1) / half * within + between) / safe)
    centred = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * draws - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :draws] / draws
    mean_var = acov[:, 0].mean(axis=0) * draws / (draws - 1)
    var_plus = mean_var * (draws - 1) / draws
    if chains > 1:
        var_plus = var_plus + x.mean(axis=1).var(axis=0, ddof=1)
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / np.where(var_plus > 0, var_plus, 1.0)
    rho[0] = 1.0
    even = rho[0:2 * (draws // 2):2] + rho[1:2 * (draws // 2):2]
    even = np.minimum.accumulate(even, axis=0) * np.cumprod(even > 0, axis=0)
    spent = np.maximum(-1.0 + 2.0 * even.sum(axis=0), 1.0 / np.log10(chains * draws))
    return rhat, chains * draws / spent


def diagnose(raw: Mapping[str, np.ndarray], exists: Mapping[str, np.ndarray], divergences: int,
             run_seconds: float) -> dict:
    """The `diagnostics` member, from draws with their chain axis. `exists[name]` masks the parameters
    that exist; a column that is constant (a fixed unit, a term that is off) is not checked. The R-hat and
    effective sample size checks name the parameter they come from as `worst`: the array and the position
    in it, as in `fitfile.DRAW_SHAPES` without the draw axis."""
    rhats, esses, labels = [], [], []
    for name in CHECKED:
        x = np.asarray(raw[name], dtype=float)
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        keep = np.broadcast_to(exists[name], x.shape[2:]).reshape(-1) if name in exists else np.ones(
            flat.shape[2], dtype=bool)
        keep = keep & np.isfinite(flat).all(axis=(0, 1)) & (flat.std(axis=(0, 1)) > 1e-12)
        if keep.any():
            rhat, ess = split_rhat_ess(flat[:, :, keep])
            rhats.append(rhat)
            esses.append(ess)
            labels += [f"{name}{[int(k) for k in np.unravel_index(n, x.shape[2:])]}" for n in np.flatnonzero(keep)]
    rhat_all = np.concatenate(rhats) if rhats else np.ones(0)
    ess_all = np.concatenate(esses) if esses else np.zeros(0)
    rhat_max = float(rhat_all.max()) if rhats else 1.0
    ess_min = float(ess_all.min()) if esses else 0.0
    worst_rhat = labels[int(rhat_all.argmax())] if labels else None
    worst_ess = labels[int(ess_all.argmin())] if labels else None
    total = int(np.asarray(raw["mu"]).shape[0] * np.asarray(raw["mu"]).shape[1])
    limit = DIVERGENT_SHARE * total
    checks = [{"name": "rhat", "value": rhat_max, "limit": RHAT_MAX, "ok": bool(rhat_max <= RHAT_MAX),
               "worst": worst_rhat},
              {"name": "ess", "value": ess_min, "limit": ESS_MIN, "ok": bool(ess_min >= ESS_MIN),
               "worst": worst_ess},
              {"name": "divergences", "value": int(divergences), "limit": limit, "ok": bool(divergences <= limit)}]
    return {"converged": all(check["ok"] for check in checks), "rhat_max": rhat_max, "ess_min": ess_min,
            "divergences": int(divergences), "run_seconds": float(run_seconds), "checks": checks}


def existing(design: M.Design) -> dict[str, np.ndarray]:
    """Per draws array, where the parameter exists. Elsewhere the fit file holds NaN."""
    m = design.masks
    has_score, has_pair = m["has_score"], m["has_pair"]
    judge_pairs = has_pair.any(axis=1)
    J, D, N = design.data["J"], design.data["D"], design.data["F"]
    return {"bias": has_score, "sigma": has_score, "tau": has_score | has_pair, "halo_target": m["halo_on"],
            "halo_call": m["halo_on"], "family": np.broadcast_to(m["model_judge"][:, None, None], (J, N, D)),
            "sd_judge_input": has_score, "sigma_pair": has_pair, "position": judge_pairs, "tie_width": judge_pairs}


def finish(design: M.Design, raw: Mapping[str, np.ndarray], keep_draws: int = 1000) -> dict[str, np.ndarray]:
    """Draws with a chain axis to the arrays of the fit file: chains joined, thinned, NaN where absent."""
    exists = existing(design)
    first = np.asarray(raw["mu"])
    total = first.shape[0] * first.shape[1]
    picks = np.arange(total) if total <= keep_draws else np.round(np.linspace(0, total - 1, keep_draws)).astype(int)
    out = {}
    for name in F.DRAW_SHAPES:
        if name not in raw:
            continue
        x = np.asarray(raw[name], dtype=np.float64)
        x = x.reshape((total,) + x.shape[2:])[picks].copy()
        if name == "halo_call":   # calls of a per_component judge are independent: the shared share is 0
            x[:, ~design.masks["panel"]] = 0.0
        if name == "sd_judge_input":
            x[:, ~design.masks["ji_on"]] = 0.0
        if name in exists:
            x[:, ~np.broadcast_to(exists[name], x.shape[1:])] = np.nan
        out[name] = x
    return out


def unpooled(design: M.Design, draws: Mapping[str, np.ndarray]) -> dict:
    """Per (run, dimension): calls per judge, and for a [0, 1] dimension the unpooled estimate, as the flow
    fit defines it: the readings less each judge's bias, weighted by 1 / noise^2, with no prior on quality."""
    def mean(name: str) -> np.ndarray:
        """The posterior mean. A parameter that does not exist is NaN in every draw, and is not read below."""
        x = np.asarray(draws[name])
        gone = np.isnan(x).all(axis=0)
        return np.where(gone, np.nan, np.where(np.isnan(x), 0.0, x).mean(axis=0))

    bias, tau, sigma = mean("bias"), mean("tau"), mean("sigma")
    dim_ids = [dim["id"] for dim in design.dims]
    out: dict[tuple, dict] = {key: {"judges": dict(calls)} for key, calls in design.cell_calls.items()}
    for (r, d), rows in design.unit_values.items():
        weights, values, calls = [], [], 0
        for j, mean, n in rows:
            noise = tau[j, d] ** 2 + sigma[j, d] ** 2 / n
            weights.append(1.0 / max(float(noise), 1e-12))
            values.append(mean - bias[j, d])
            calls += n
        total = float(np.sum(weights))
        out[(design.runs[r]["run"], dim_ids[d])]["unpooled"] = {
            "mean": float(np.dot(weights, values) / total), "sd": float(np.sqrt(1.0 / total)), "n": int(calls)}
    return out


def fit_design(design: M.Design, *, seed: int = 0, draws: int | None = None, chains: int | None = None,
               tune: int | None = None, cores: int | None = None, keep_draws: int = 1000,
               inputs: dict | None = None, options: dict | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Sample a design and assemble the fit: `(every member of fit.json but draws, the draws)`."""
    from . import engine_pymc as engine

    draws = DEFAULT_DRAWS if draws is None else int(draws)
    chains = DEFAULT_CHAINS if chains is None else int(chains)
    result = engine.sample(design.data, seed=seed, draws=draws, tune=tune, chains=chains, cores=cores)
    raw = result["draws"]
    checked = {**existing(design), "halo_call": design.masks["panel"]}   # elsewhere it is 0 by definition
    diagnostics = diagnose(raw, checked, result["divergences"], result["run_seconds"])
    final = finish(design, raw, keep_draws)
    full_options = {**design.options, "draws": draws, "tune": draws if tune is None else int(tune),
                    "chains": chains, "keep_draws": int(keep_draws), **(options or {})}
    meta = F.assemble(rubric=design.rubric, judges=design.judges, runs=design.runs, draws=final,
                      terms=design.terms, anchor=design.anchor, diagnostics=diagnostics, inputs=inputs,
                      options=full_options, engine={"name": engine.NAME, "version": result["version"]}, seed=seed,
                      components=design.components, counts=design.counts, judge_counts=design.judge_counts,
                      readings=unpooled(design, final), notes=design.notes, relative=design.relative)
    return meta, final


def _positive(value: Any, option: str, least: int = 1) -> None:
    if value is not None and value < least:
        raise C.ContractError("eval fit", option, f"{value} is below {least}", f"give {least} or more", "bad_option")


def run(args: argparse.Namespace) -> int:
    """`eval fit`."""
    from . import engine_pymc as engine

    _positive(args.draws, "--draws", 20)
    _positive(args.chains, "--chains")
    _positive(args.keep_draws, "--keep-draws")
    rubric = C.read_document(args.rubric, C.SCHEMA_RUBRIC)
    judges = C.read_document(args.judges, C.SCHEMA_JUDGES)
    study = C.read_study(args.study)
    design = M.build(study, rubric, judges, components=args.components, structure=args.structure, halo=args.halo,
                     cuts=args.cuts, terms=args.terms, allow_simulated=args.allow_simulated)
    lines = M.report_lines(design)
    if args.check_only:
        if args.json:
            emit_json(OUTPUT_SCHEMA, {"checked_only": True, "fit": None, **M.report(design)})
        else:
            print("\n".join(lines))
        return EXIT_OK
    if not engine.available():
        print(engine.INSTALL, file=sys.stderr)
        return EXIT_USER
    print("\n".join(lines), file=sys.stderr if args.json else sys.stdout, flush=True)
    out = Path(args.out) if args.out else Path(args.study) / "fit.json"
    inputs = C.input_digests({**study.files(), "rubric": args.rubric, "judges": args.judges})
    meta, draws = fit_design(design, seed=args.seed, draws=args.draws, chains=args.chains,
                             keep_draws=args.keep_draws, inputs=inputs,
                             options={"allow_simulated": bool(args.allow_simulated)})
    path = F.write_fit(out, meta, draws)
    diagnostics = meta["diagnostics"]
    if args.json:
        emit_json(OUTPUT_SCHEMA, {"checked_only": False, "fit": str(path), "draws": str(F.draws_path(path)),
                                  "diagnostics": diagnostics, **M.report(design)})
    else:
        print(f"fit: {path} ({next(iter(draws.values())).shape[0]} draws in {F.draws_path(path).name})")
        print(f"converged: {'yes' if diagnostics['converged'] else 'no'} (R-hat {diagnostics['rhat_max']:.3f}, "
              f"effective sample size {diagnostics['ess_min']:.0f}, {diagnostics['divergences']} divergences, "
              f"{diagnostics['run_seconds']:.0f} s)")
        if not diagnostics["converged"]:
            print("the fit did not converge: later commands refuse it without --allow-unconverged; "
                  "try more --draws")
    return EXIT_OK
