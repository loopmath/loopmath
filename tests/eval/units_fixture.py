"""Two fit files with the same draws, one in the engine's unit and one in the reported unit, for the
invariance tests of `units.py` (synthetic data only).

`pair_of_fits` builds the first with `testing.build_fit` on the example rubric, with `overall` made a
pairs-only dimension compared by the reviewer and the cheap judge, and the reviewer's tau on the binary
dimension large and uncertain, so the unit factor differs from draw to draw. The second holds the same
draws put in the reported unit by `units.convert`, with the factor the engine's rule gives (the reviewer
carries the unit of both dimensions), and its `unit` records. `write_pair` does the same for any draws
and any factor, for the counterexamples to a reader that would decide in the reported unit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import testing
from loopmath.eval import units as U

PAIRED = ("reviewer", "cheap")


def _terms(judges: dict, rubric: dict, relative: str) -> list[dict]:
    """The default identification report of `testing.build_fit`, with the pair terms of the paired judges
    identified and the reviewer's judge-by-input term on and identified on every dimension."""
    rows = []
    dims = C.dimensions(rubric)
    for row in judges["judges"]:
        jid = row["id"]
        for dim in dims:
            for name in ("sigma", "tau"):
                rows.append(F.term(name, "identified", judge=jid, dimension=dim["id"], count=2))
            if jid == "reviewer":
                rows.append(F.term("judge_input", "identified", judge=jid, dimension=dim["id"], count=12))
        if row["kind"] == "model":
            rows.append(F.term("family", "identified", judge=jid, count=2))
        else:
            rows.append(F.term("family", "absent", judge=jid, reason="not a model judge"))
        for name in ("position", "sigma_pair", "tie_width"):
            if jid in PAIRED:
                rows.append(F.term(name, "identified", judge=jid, count=24))
            else:
                rows.append(F.term(name, "absent", judge=jid, reason="no pairs"))
        rows.append(F.term("halo_split", "prior_only", judge=jid))
        if jid != "reviewer":
            rows.append(F.term("judge_input", "off", judge=jid))
        rows.append(F.term("reference_family", "absent", judge=jid, reason="no reference is shown"))
    rows.append(F.term("levels_split", "identified", count=3))
    for dim in dims:
        rows.append(F.term("anchor_bias", "identified", dimension=dim["id"], count=36))
        if dim["scale"]["kind"] == "binary":
            rows.append(F.term("binary_status", "identified", dimension=dim["id"], count=36))
    return rows


def pair_of_fits(folder: Path, *, posterior_sd: float = 0.3, n_draws: int = 400, seed: int = 4):
    """`(engine fit, reported fit, unit[S, D])`, both `fitfile.Fit`, written under `folder`."""
    rubric, judges = testing.example_rubric(), testing.example_judges()
    runs = testing.example_runs()
    index = F.build_index(rubric, judges, runs)
    dims = C.dimensions(rubric)
    ids = [dim["id"] for dim in dims]
    b, o = ids.index("final/numbers_right"), ids.index("final/overall")
    truth = testing.default_truth(index, dims, judges)
    J, D = len(index["judges"]), len(dims)
    truth["tau"][0, b] = 0.7                               # the reviewer: a large judge-by-output part
    truth["sd_judge_input"][0, b] = 0.4                    # and a judge-by-input part
    truth["mu"][:, o] = [-0.5, 0.1, 0.4]                   # overall is relative: arm means on a probit-like scale
    truth["sd_input"][o], truth["sd_run"][o] = 0.5, 0.4
    truth["sigma_pair"] = np.full((J, D), np.nan)
    truth["sigma_pair"][0, o], truth["sigma_pair"][2, o] = 1.0, 1.4
    truth["tau"][0, o], truth["tau"][2, o] = 0.5, 0.8
    truth["position"] = np.array([0.2, np.nan, 0.5])
    truth["tie_width"] = np.array([0.4, np.nan, 0.3])
    relative = ["final/overall"]
    built = testing.build_fit(folder / "engine" / "fit.json", rubric=rubric, judges=judges, runs=runs, truth=truth,
                              n_draws=n_draws, posterior_sd=posterior_sd, seed=seed, relative=relative,
                              terms=_terms(judges, rubric, "final/overall"))
    # The engine's fit as an older checkout wrote it: schema 1, no unit records, no unit draws.
    old = C.load_json(built.path)
    old["schema"] = "loopmath.eval-fit/1"
    for dim in old["dimensions"]:
        dim.pop("unit")
    C.write_text_atomic(built.path, C.dumps(old))
    engine = F.read_fit(built.path)
    draws = {name: np.array(x) for name, x in engine.draws.items() if name != "unit"}
    sigma_w, pair_w = np.zeros((J, D)), np.zeros((J, D))
    sigma_w[0, b], pair_w[0, o] = 1.0, 1.0                 # the reviewer anchors both
    binary = np.array([dim["scale"]["kind"] == "binary" for dim in dims])
    pairs_only = np.array([dim_id in relative for dim_id in ids])
    ji_on = np.zeros((J, D), dtype=bool)
    ji_on[0] = True
    unit = U.factor(draws, sigma_w, pair_w, binary, pairs_only, ji_on)
    converted = U.convert(draws, unit)
    who = [["reviewer"] if d in (b, o) else None for d in range(D)]
    rebuilt = _assemble(engine.meta, rubric, judges, runs, converted, relative, U.records(dims, relative, who, unit))
    path = F.write_fit(folder / "reported" / "fit.json", rebuilt, converted)
    return engine, F.read_fit(path), unit


def _assemble(meta: dict, rubric: dict, judges: dict, runs: list, draws: dict, relative, units) -> dict:
    """Every member of `fit.json` but `draws`, for `draws`, with the records of the fit `meta`."""
    return F.assemble(rubric=rubric, judges=judges, runs=runs, draws=draws, terms=meta["terms"],
                      anchor=meta["anchor"], diagnostics=meta["diagnostics"], engine=meta["engine"],
                      seed=meta["seed"], options=meta["options"], counts=meta["counts"],
                      judge_counts={(row["id"], d["dimension"]): {k: d[k] for k in ("n_targets", "n_calls", "n_pairs")}
                                    for row in meta["judges"] for d in row["dims"]},
                      readings={(t["run"], d["dimension"]): {"judges": d["judges"],
                                                              **({"unpooled": d["unpooled"]} if d["unpooled"] else {})}
                                for t in meta["targets"] for d in t["dims"]},
                      relative=list(relative), units=units)


def write_pair(folder: Path, meta: dict, draws: dict, unit, *, rubric: dict | None = None, judges: dict | None = None,
               runs: list | None = None, relative=("final/overall",)) -> tuple[F.Fit, F.Fit]:
    """`(schema 1 fit, schema 2 fit)` of the same `draws`, given in the engine's unit, written under `folder`.

    The second divides them by `unit[S, D]` (`units.convert`), which must be 1 on the dimensions the data
    fix and may be any positive number per draw elsewhere: whatever the factor, a decision must not move.
    `meta` is a fit's `fit.json` whose records (terms, anchor, counts) both files take."""
    rubric, judges = rubric or testing.example_rubric(), judges or testing.example_judges()
    runs = runs if runs is not None else testing.example_runs()
    dims = C.dimensions(rubric)
    draws = {name: np.array(x, dtype=float) for name, x in draws.items() if name != "unit"}
    old = F.write_fit(folder / "engine" / "fit.json", _assemble(meta, rubric, judges, runs, draws, relative, None),
                      draws)
    obj = C.load_json(old)
    obj["schema"] = "loopmath.eval-fit/1"
    for dim in obj["dimensions"]:
        dim.pop("unit")
    C.write_text_atomic(old, C.dumps(obj))
    converted = U.convert(draws, unit)
    anchors = list(meta["anchor"].get("judges") or [])
    who = [anchors if U.unit_of(dim["scale"]["kind"], dim["id"] in relative) in ("total_noise", "total_pair_noise")
           else None for dim in dims]
    new = F.write_fit(folder / "reported" / "fit.json",
                      _assemble(meta, rubric, judges, runs, converted, relative, U.records(dims, relative, who, unit)),
                      converted)
    return F.read_fit(old), F.read_fit(new)
