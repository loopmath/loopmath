"""The design of a fit: rows to arrays and the identification report (pure, no engine).

Not a command: `fit.py` calls `build`, prints `report_lines`, and hands `Design.data` to the engine.

What `build` does
-----------------
1. Reads the live readings and pairs of a study (a replaced row is not read) and lists every row it
   does not use in `notes`, with the reason. Item lines of one call are summed to k of n.
2. Maps every reading to the fitting scale (`quality.fitting_value` and friends) and to positions in
   the axes of `fitfile.build_index`.
3. Works out the identification report of the model (`terms`), the anchor state per dimension, and
   which optional terms are on.
4. Lays the rows out for the sampler. The layout sums out what can be summed out exactly:

   - A cell is one (judge, run, dimension). A `linear` or `fraction` cell read by `per_component` calls
     goes in as its sufficient statistics: the number of calls, their mean and their sum of squares.
     Its judge-by-output term and its call noise are summed out.
   - A `binary` or `levels` cell with repeats that no pair looks at goes in as a group: the engine sums
     its judge-by-output term out by quadrature.
   - Every other reading goes in one by one. A cell with one reading and no pair has its
     judge-by-output term summed into the reading's noise. Any other cell keeps the term as an
     explicit unknown; so does every cell a pair looks at.
   - Every call of a `panel` judge gets one shared unknown for the shared part of its call noise.
   - An `items` reading of more than one item keeps its call noise as an explicit unknown, because the
     binomial does not sum it out.

Conventions
-----------
- Thresholds: a count below `WEAK_BELOW` gives the state `weak`; 0 gives `prior_only`.
- `sigma` of a `binary` dimension has no unit of its own. The anchors' mean log `sigma` is 0 (the
  judges' mean when no anchor read the dimension). A single judge that carries the unit has `sigma`
  fixed at 1 and the term state `off`.
- A dimension with pairs and no readings is `relative`: arm means sum to 0, the unit is the anchors'
  `sigma_pair`, and bias and `sigma` do not exist there.
- `family` is on (under `auto`) for a model judge when judges from 2 or more families have readings and
  the judge read outputs from 2 or more families. `judge_input` is on for a judge and dimension with
  2 or more runs per input on 10 or more inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import contracts as C
from . import fitfile as F
from . import quality as Q
from .commands import not_built
from .contracts import ContractError

LINEAR, BINARY, LEVELS, ITEMS = 0, 1, 2, 3
KIND_CODE = {"linear": LINEAR, "fraction": LINEAR, "binary": BINARY, "levels": LEVELS, "items": ITEMS}
OPTIONAL_TERMS = ("family", "judge_input")
TERM_SWITCHES = ("on", "off", "auto")
WEAK_BELOW = 5            # a count from 1 to 4 is "weak"
JUDGE_INPUT_NEEDS = 10    # inputs with 2 or more runs read by the judge
FULL_UP_TO = 8            # --structure default: full up to 8 dimensions, factor:2 above
OPEN = 1.0e6              # the outer cut points of a levels scale: the two outer intervals are open
UNIT_SCALE, PROBIT_SCALE = 0.25, 1.0   # prior scale of sds on [0, 1] dimensions and on probit dimensions


def run(args: argparse.Namespace) -> int:
    """Not a command. `eval fit` calls `build`."""
    return not_built(args)


@dataclass
class Design:
    """Everything `fit.py` needs: the arrays for the engine and the records for the fit file."""

    rubric: dict
    judges: dict
    runs: list
    components: list | None
    index: dict
    dims: list
    options: dict
    data: dict
    terms: list
    anchor: dict
    notes: list
    counts: dict
    judge_counts: dict
    relative: list
    cell_calls: dict = field(default_factory=dict)    # (run id, dimension id) -> {judge id: calls}
    unit_values: dict = field(default_factory=dict)   # (run position, d) -> [(judge position, mean, calls)]
    masks: dict = field(default_factory=dict)

    @property
    def sizes(self) -> dict:
        return {name: self.data[name] for name in ("A", "D", "J", "F", "I", "R")}


def parse_terms(values: Sequence[str] | None) -> dict:
    """`--terms NAME=on|off|auto` (repeatable) as a mapping. The default of every optional term is `auto`."""
    out = {name: "auto" for name in OPTIONAL_TERMS}
    for text in values or []:
        name, sep, state = str(text).partition("=")
        if not sep or name not in OPTIONAL_TERMS or state not in TERM_SWITCHES:
            raise ContractError("eval fit", f"--terms {text}", "not an optional term and a switch",
                                "write family=on|off|auto or judge_input=on|off|auto", "bad_option")
        out[name] = state
    return out


def parse_structure(text: str | None, n_dims: int) -> tuple[str, int]:
    """`--structure` as `(form, K)`: `full`, `diag` or `factor` with K factors. The default follows D."""
    if text is None:
        return ("full", 0) if n_dims <= FULL_UP_TO else ("factor", min(2, n_dims))
    if text in ("full", "diag"):
        return (text, 0)
    name, sep, k = text.partition(":")
    if name == "factor" and sep and k.isdigit() and 1 <= int(k) <= max(1, n_dims):
        return ("factor", int(k))
    raise ContractError("eval fit", f"--structure {text}", "not a covariance form",
                        f"write full, diag or factor:K with K from 1 to {max(1, n_dims)}", "bad_option")


def _state(count: int, enough: int = WEAK_BELOW) -> str:
    return "prior_only" if count <= 0 else ("weak" if count < enough else "identified")


def _note(kind: str, row_id: str, reason: str) -> dict:
    return {"kind": kind, "id": row_id, "reason": reason}


def _same_setting(a: dict, b: dict) -> bool:
    """Two judges that differ only in `mode`: the same model, effort, prompt and reference."""
    keys = ("kind", "model", "effort", "prompt", "reference")
    return a["id"] != b["id"] and a["mode"] != b["mode"] and all(a.get(k) == b.get(k) for k in keys)


def build(study: C.Study, rubric: dict, judges: dict, *, components: Sequence[str] | None = None,
          structure: str | None = None, halo: str = "factor", cuts: str = "fixed",
          terms: Sequence[str] | None = None, allow_simulated: bool = False) -> Design:
    """Turn a study into a `Design`. Refusals are `ContractError`s; a row that is not used is a note."""
    if halo != "factor":
        raise ContractError("eval fit", "--halo full", "the full halo form is not built yet",
                            "use --halo factor (the default)", "not_built")
    if cuts != "fixed":
        raise ContractError("eval fit", "--cuts free", "free cut points are not built yet",
                            "use --cuts fixed (the default)", "not_built")
    switches = parse_terms(terms)
    if components is not None:
        for name in components:
            if C.component(rubric, name) is None:
                raise ContractError("eval fit", f"--components {name}", f'unknown component "{name}"',
                                    "name components of the rubric", "unknown_component")
    runs = list(study.runs)
    if not runs:
        raise ContractError(C.RUNS_FILE, "", "the study has no runs", "import runs first", "empty_study")
    for run in runs:
        if run["simulated"] and not allow_simulated:
            raise ContractError(C.RUNS_FILE, f'run "{run["run"]}"', "the run is simulated",
                                "pass --allow-simulated to fit a simulated study", "simulated")
    dims = C.dimensions(rubric, components)
    if not dims:
        raise ContractError("rubric.json", "components", "no dimension to fit",
                            "name components that apply to a step of the rubric", "no_dimensions")
    index = F.build_index(rubric, judges, runs, components)
    form = parse_structure(structure, len(dims))
    A, D, J, N, I, R = (len(index["arms"]), len(dims), len(index["judges"]), len(index["families"]),
                        len(index["inputs"]), len(runs))
    run_pos = {run["run"]: r for r, run in enumerate(runs)}
    judge_pos = {name: j for j, name in enumerate(index["judges"])}
    dim_pos = {(dim["step"], dim["component"]): d for d, dim in enumerate(dims)}
    run_input = np.asarray(index["run_input"], dtype=np.int64)
    family_pos = {name: f for f, name in enumerate(index["families"])}
    declared = judges["judges"]
    ref = C.rubric_ref(rubric)
    notes: list[dict] = []

    # ---- readings ------------------------------------------------------------------------------
    live, dropped = C.live(study.scores, C.SCORES_FILE)
    notes += [_note("reading", row["id"], f'replaced by {row["by"]}') for row in dropped]
    readings: list[dict] = []          # one per fitted reading (item lines already summed)
    item_calls: dict[tuple, dict] = {}
    for row in live:
        comp = C.component(rubric, row["component"])
        reason = None
        if row["run"] not in run_pos:
            reason = f'unknown run "{row["run"]}"'
        elif row["judge"] not in judge_pos:
            reason = f'unknown judge "{row["judge"]}"'
        elif comp is None:
            reason = f'unknown component "{row["component"]}"'
        elif (row["step"], row["component"]) not in dim_pos:
            reason = "its step and component are not a fitted dimension"
        elif row["rubric"] != ref:
            reason = f'scored under rubric {row["rubric"]}, not {ref}'
        else:
            try:
                C.check_reading(row, comp)
            except ContractError as exc:
                reason = exc.problem
        if reason is not None:
            notes.append(_note("reading", row["id"], reason))
            continue
        d = dim_pos[(row["step"], row["component"])]
        dim = dims[d]
        base = {"r": run_pos[row["run"]], "d": d, "j": judge_pos[row["judge"]], "call": row["call"],
                "kind": KIND_CODE[dim["scale"]["kind"]], "y": 0.0, "lo": -OPEN, "hi": OPEN, "k": 0, "n": 0,
                "ids": [row["id"]]}
        kind = dim["scale"]["kind"]
        if kind in ("linear", "fraction"):
            base["y"] = float(Q.fitting_value(dim, row["value"]))
        elif kind == "binary":   # a pass is the open interval above 0, a fail the one below
            base["y"] = float(Q.fitting_value(dim, row["value"]))
            base["lo"], base["hi"] = (0.0, OPEN) if base["y"] >= 0.5 else (-OPEN, 0.0)
        elif kind == "levels":
            level = Q.fitting_level(dim, row["value"])
            fixed = Q.fixed_cuts(len(dim["scale"]["levels"]))
            base["y"] = level / (len(dim["scale"]["levels"]) - 1)
            base["lo"] = float(fixed[level - 1]) if level > 0 else -OPEN
            base["hi"] = float(fixed[level]) if level < len(fixed) else OPEN
        elif "of" in row:
            base["k"], base["n"] = Q.fitting_count(dim, int(row["value"]), int(row["of"]))
        else:  # an item line: summed per call below
            key = (base["j"], row["call"], base["r"], d)
            entry = item_calls.setdefault(key, {**base, "ids": [], "passed": 0, "lines": 0})
            entry["ids"].append(row["id"])
            entry["passed"] += int(row["value"])
            entry["lines"] += 1
            continue
        readings.append(base)
    counted = {(x["j"], x["call"], x["r"], x["d"]) for x in readings if x["kind"] == ITEMS}
    for key, entry in item_calls.items():
        if key in counted:
            notes += [_note("reading", i, "its call also has a count line: the count line is read")
                      for i in entry["ids"]]
            continue
        entry["k"], entry["n"] = Q.fitting_count(dims[entry["d"]], entry.pop("passed"), entry.pop("lines"))
        readings.append(entry)
    n_reading_rows = sum(len(x["ids"]) for x in readings)

    # ---- pairs ---------------------------------------------------------------------------------
    live_pairs, dropped_pairs = C.live(study.pairs, C.PAIRS_FILE)
    notes += [_note("pair", row["id"], f'replaced by {row["by"]}') for row in dropped_pairs]
    pairs: list[dict] = []
    for row in live_pairs:
        comp_id = row["component"] or C.overall_component(rubric, row["step"])
        reason = None
        if row["a"] not in run_pos or row["b"] not in run_pos:
            reason = "it names an unknown run"
        elif row["judge"] not in judge_pos:
            reason = f'unknown judge "{row["judge"]}"'
        elif comp_id is None:
            reason = f'the step "{row["step"]}" has no overall component and the pair names none'
        elif (row["step"], comp_id) not in dim_pos:
            reason = "its step and component are not a fitted dimension"
        elif row["rubric"] != ref:
            reason = f'scored under rubric {row["rubric"]}, not {ref}'
        elif runs[run_pos[row["a"]]]["input"] != runs[run_pos[row["b"]]]["input"]:
            reason = "its two runs have different inputs"
        if reason is not None:
            notes.append(_note("pair", row["id"], reason))
            continue
        o = 0.0 if row["shown_first"] is None else (1.0 if row["shown_first"] == "a" else -1.0)
        first = None if row["shown_first"] is None else row[row["shown_first"]]
        pairs.append({"ra": run_pos[row["a"]], "rb": run_pos[row["b"]], "d": dim_pos[(row["step"], comp_id)],
                      "j": judge_pos[row["judge"]], "o": o, "v": {"a": 0, "tie": 1, "b": 2}[row["verdict"]],
                      "first": first, "id": row["id"]})
    if not readings and not pairs:
        raise ContractError(str(study.folder), "", "no reading and no pair can be fitted",
                            "see eval scores check for what is wrong with the rows", "nothing_to_fit")

    # ---- what was read by whom -----------------------------------------------------------------
    has_score = np.zeros((J, D), dtype=bool)
    has_pair = np.zeros((J, D), dtype=bool)
    cells: dict[tuple, list[int]] = {}
    for n, x in enumerate(readings):
        has_score[x["j"], x["d"]] = True
        cells.setdefault((x["j"], x["r"], x["d"]), []).append(n)
    pair_cells = set()
    for p in pairs:
        has_pair[p["j"], p["d"]] = True
        pair_cells.update({(p["j"], p["ra"], p["d"]), (p["j"], p["rb"], p["d"])})
    pairs_only = has_pair.any(axis=0) & ~has_score.any(axis=0)
    kinds = [dim["scale"]["kind"] for dim in dims]
    probit = np.array([kind in Q.PROBIT_KINDS for kind in kinds])
    scale = np.where(probit | pairs_only, PROBIT_SCALE, UNIT_SCALE)
    mu_loc = np.where(probit | pairs_only, 0.0, 0.5)
    person = np.array([row["kind"] == "person" for row in declared])
    model_judge = np.array([row["kind"] == "model" for row in declared])
    panel = np.array([row["mode"] == "panel" for row in declared])
    anchors = [judge_pos[name] for name in judges["anchor"]]

    # halo: a judge that reads 2 or more dimensions of one target has a shared factor per target
    step_of = [dim["step"] for dim in dims]
    seen_dims: dict[tuple, set] = {}
    for (j, r, d) in list(cells) + sorted(pair_cells):
        seen_dims.setdefault((j, r, step_of[d]), set()).add(d)
    halo_on = np.zeros(J, dtype=bool)
    for (j, _, _), found in seen_dims.items():
        if len(found) >= 2:
            halo_on[j] = True
    jt_pos = {key: n for n, key in enumerate(sorted(k for k in seen_dims if halo_on[k[0]]))}

    # ---- counts behind the identification report -------------------------------------------------
    targets_by = [[set() for _ in range(D)] for _ in range(J)]       # targets read by judge j on d
    repeats_by = np.zeros((J, D), dtype=int)                          # targets with 2 or more calls
    for (j, r, d), rows in cells.items():
        targets_by[j][d].add(r)
        repeats_by[j, d] += len(rows) >= 2
    judges_on: list[dict] = [dict() for _ in range(D)]                # d: run -> set of judges
    for (j, r, d) in cells:
        judges_on[d].setdefault(r, set()).add(j)
    shared_by = np.zeros((J, D), dtype=int)
    for d in range(D):
        for r, who in judges_on[d].items():
            if len(who) >= 2:
                for j in who:
                    shared_by[j, d] += 1
    anchor_targets = [len(set().union(*[targets_by[j][d] for j in anchors])) if anchors else 0 for d in range(D)]
    need = judges["min_anchor_targets"]
    anchored = np.array([anchor_targets[d] >= need for d in range(D)])

    # bias: the anchors (or, unanchored, the judges with readings) sum to 0 per dimension
    bias_w = np.zeros((J, D))
    sigma_w = np.zeros((J, D))
    pair_w = np.zeros((J, D))
    for d in range(D):
        scored = [j for j in range(J) if has_score[j, d]]
        anchor_scored = [j for j in anchors if has_score[j, d]]
        members = anchor_scored if anchored[d] else scored
        for j in members:
            bias_w[j, d] = 1.0 / len(members)
        unit = anchor_scored or scored
        for j in unit:
            sigma_w[j, d] = 1.0 / len(unit)
        with_pairs = [j for j in range(J) if has_pair[j, d]]
        pair_unit = [j for j in anchors if has_pair[j, d]] or with_pairs
        for j in pair_unit:
            pair_w[j, d] = 1.0 / len(pair_unit)
    sigma_fixed = np.array([kinds[d] == "binary" for d in range(D)]) & has_score.any(axis=0)

    # the family term
    out_families: list[set] = [set() for _ in range(J)]
    for (j, r, d) in list(cells) + sorted(pair_cells):
        out_families[j].add(F.target_family(runs[r], step_of[d]))
    judge_families = {declared[j]["model"]["family"] for j in range(J)
                      if model_judge[j] and (has_score[j].any() or has_pair[j].any())}
    family_ok = np.array([bool(model_judge[j]) and len(judge_families) >= 2 and len(out_families[j]) >= 2
                          for j in range(J)])
    if switches["family"] == "auto":
        family_on = model_judge & family_ok
    else:
        family_on = model_judge & (switches["family"] == "on") & np.array([len(x) >= 1 for x in out_families])
    family_on = family_on & (N >= 2)

    # the judge-by-input term
    ji_inputs = np.zeros((J, D), dtype=int)
    per_input: dict[tuple, int] = {}
    for (j, r, d) in cells:
        per_input[(j, int(run_input[r]), d)] = per_input.get((j, int(run_input[r]), d), 0) + 1
    for (j, _, d), n in per_input.items():
        ji_inputs[j, d] += n >= 2
    if switches["judge_input"] == "auto":
        ji_on = ji_inputs >= JUDGE_INPUT_NEEDS
    else:
        ji_on = has_score & (switches["judge_input"] == "on")
    ji_pos = {key: n for n, key in enumerate(sorted(k for k in per_input if ji_on[k[0], k[2]]))}

    # ---- the layout for the engine ---------------------------------------------------------------
    calls_in: dict[tuple, int] = {}
    for x in readings:
        if panel[x["j"]]:
            calls_in[(x["j"], x["call"])] = calls_in.get((x["j"], x["call"]), 0) + 1
    pc_pos = {key: n for n, key in enumerate(sorted(calls_in))}

    def family_of(r: int, d: int) -> int:
        return family_pos[F.target_family(runs[r], step_of[d])]

    suff: dict[str, list] = {name: [] for name in ("r", "d", "j", "f", "jt", "ji", "n", "ybar", "ss")}
    suff_pos: dict[tuple, int] = {}
    quad: dict[str, list] = {name: [] for name in ("r", "d", "j", "f", "jt", "ji", "pc", "lo", "hi")}
    quad_groups: dict[int, list] = {}   # calls per cell -> rows of positions in `quad`
    gen: dict[str, list] = {name: [] for name in ("kind", "r", "d", "j", "f", "jt", "ji", "cell", "pc", "e", "y",
                                                  "lo", "hi", "k", "n")}
    explicit: dict[tuple, int] = {}
    n_e = 0
    for key in sorted(cells):
        j, r, d = key
        rows = [readings[n] for n in cells[key]]
        jt = jt_pos.get((j, r, step_of[d]), -1)
        ji = ji_pos.get((j, int(run_input[r]), d), -1)
        f = family_of(r, d)
        shared_call = any((j, x["call"]) in pc_pos for x in rows)
        if rows[0]["kind"] == LINEAR and not shared_call:
            suff_pos[key] = len(suff["r"])
            y = np.array([x["y"] for x in rows])
            for name, value in (("r", r), ("d", d), ("j", j), ("f", f), ("jt", jt), ("ji", ji), ("n", len(rows)),
                                ("ybar", float(y.mean())), ("ss", float(((y - y.mean()) ** 2).sum()))):
                suff[name].append(value)
            continue
        if rows[0]["kind"] in (BINARY, LEVELS) and len(rows) >= 2 and key not in pair_cells:
            start = len(quad["r"])
            for x in rows:
                for name, value in (("r", r), ("d", d), ("j", j), ("f", f), ("jt", jt), ("ji", ji),
                                    ("pc", pc_pos.get((j, x["call"]), -1)), ("lo", x["lo"]), ("hi", x["hi"])):
                    quad[name].append(value)
            quad_groups.setdefault(len(rows), []).append(list(range(start, start + len(rows))))
            continue
        cell = -1
        if len(rows) >= 2 or key in pair_cells:
            cell = explicit.setdefault(key, len(explicit))
        for x in rows:
            e = -1
            if x["kind"] == ITEMS and x["n"] > 1:
                e, n_e = n_e, n_e + 1
            for name, value in (("kind", x["kind"]), ("r", r), ("d", d), ("j", j), ("f", f), ("jt", jt), ("ji", ji),
                                ("cell", cell), ("pc", pc_pos.get((j, x["call"]), -1)), ("e", e), ("y", x["y"]),
                                ("lo", x["lo"]), ("hi", x["hi"]), ("k", x["k"]), ("n", x["n"])):
                gen[name].append(value)
    for key in sorted(pair_cells):
        explicit.setdefault(key, len(explicit))
    pair: dict[str, list] = {name: [] for name in ("ra", "rb", "d", "j", "fa", "fb", "ca", "cb", "jta", "jtb", "o",
                                                   "v")}
    for p in pairs:
        j, d = p["j"], p["d"]
        for name, value in (("ra", p["ra"]), ("rb", p["rb"]), ("d", d), ("j", j), ("fa", family_of(p["ra"], d)),
                            ("fb", family_of(p["rb"], d)), ("ca", explicit[(j, p["ra"], d)]),
                            ("cb", explicit[(j, p["rb"], d)]), ("jta", jt_pos.get((j, p["ra"], step_of[d]), -1)),
                            ("jtb", jt_pos.get((j, p["rb"], step_of[d]), -1)), ("o", p["o"]), ("v", p["v"])):
            pair[name].append(value)
    cell_keys = sorted(explicit, key=explicit.get)  # type: ignore[arg-type]
    ji_keys = sorted(ji_pos, key=ji_pos.get)  # type: ignore[arg-type]
    pc_keys = sorted(pc_pos, key=pc_pos.get)  # type: ignore[arg-type]
    pair_scale = np.array([scale[has_pair[j]].mean() if has_pair[j].any() else 1.0 for j in range(J)])

    def ints(values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    data: dict[str, Any] = {
        "A": A, "D": D, "J": J, "F": N, "I": I, "R": R, "structure": form[0], "K": form[1],
        "run_arm": ints(index["run_arm"]), "run_input": run_input, "scale": scale, "mu_loc": mu_loc,
        "pairs_only": pairs_only, "person": person, "sigma_fixed": sigma_fixed,
        "unit_dim": ~(probit | pairs_only), "bias_w": bias_w, "sigma_w": sigma_w, "pair_w": pair_w,
        "halo_on": halo_on, "panel": panel, "family_on": family_on, "ji_on": ji_on, "pair_scale": pair_scale,
        "n_jt": len(jt_pos), "n_ji": len(ji_pos), "n_cell": len(explicit), "n_pc": len(pc_pos), "n_e": n_e,
        "cell_j": ints([k[0] for k in cell_keys]), "cell_d": ints([k[2] for k in cell_keys]),
        "cell_suff": ints([suff_pos.get(k, -1) for k in cell_keys]),
        "person_tight": person[:, None] & np.array([kind in ("linear", "fraction") for kind in kinds])[None, :],
        "k_groups": [ints(quad_groups[n]) for n in sorted(quad_groups)],
        "levels_dim": np.array([kind == "levels" for kind in kinds]),
        "ji_j": ints([k[0] for k in ji_keys]), "ji_d": ints([k[2] for k in ji_keys]),
        "pc_j": ints([k[0] for k in pc_keys]),
    }
    for name in ("r", "d", "j", "f", "jt", "ji", "n"):
        data[f"s_{name}"] = ints(suff[name])
    data["s_ybar"], data["s_ss"] = np.asarray(suff["ybar"], dtype=float), np.asarray(suff["ss"], dtype=float)
    for name in ("kind", "r", "d", "j", "f", "jt", "ji", "cell", "pc", "e", "k", "n"):
        data[f"g_{name}"] = ints(gen[name])
    for name in ("y", "lo", "hi"):
        data[f"g_{name}"] = np.asarray(gen[name], dtype=float)
    for name in ("ra", "rb", "d", "j", "fa", "fb", "ca", "cb", "jta", "jtb", "v"):
        data[f"p_{name}"] = ints(pair[name])
    data["p_o"] = np.asarray(pair["o"], dtype=float)
    for name in ("r", "d", "j", "f", "jt", "ji", "pc"):
        data[f"k_{name}"] = ints(quad[name])
    for name in ("lo", "hi"):
        data[f"k_{name}"] = np.asarray(quad[name], dtype=float)

    # ---- the identification report ---------------------------------------------------------------
    terms: list[dict] = []
    ids = index["judges"]
    dim_ids = [dim["id"] for dim in dims]
    tau_state = np.full((J, D), "absent", dtype=object)
    for j in range(J):
        for d in range(D):
            if not has_score[j, d]:
                why = ("the judge has pairs only on this dimension" if has_pair[j, d]
                       else "the judge has no readings on this dimension")
                terms.append(F.term("sigma", "absent", judge=ids[j], dimension=dim_ids[d], reason=why))
                if has_pair[j, d]:
                    tau_state[j, d] = "prior_only"
                    terms.append(F.term("tau", "prior_only", judge=ids[j], dimension=dim_ids[d]))
                else:
                    terms.append(F.term("tau", "absent", judge=ids[j], dimension=dim_ids[d], reason=why))
                continue
            count = int(repeats_by[j, d])
            if sigma_fixed[d] and sigma_w[j, d] == 1.0:
                terms.append(F.term("sigma", "off", judge=ids[j], dimension=dim_ids[d], count=count,
                                    reason="the unit of a binary dimension: fixed at 1"))
            else:
                terms.append(F.term("sigma", _state(count), judge=ids[j], dimension=dim_ids[d], count=count))
            both = min(count, int(shared_by[j, d]))
            tau_state[j, d] = _state(both)
            terms.append(F.term("tau", tau_state[j, d], judge=ids[j], dimension=dim_ids[d], count=both))
    runs_per_input: dict[int, int] = {}
    for i in run_input:
        runs_per_input[int(i)] = runs_per_input.get(int(i), 0) + 1
    split = sum(1 for n in runs_per_input.values() if n >= 2)
    terms.append(F.term("levels_split", _state(split), count=split))
    anchor_rows = []
    for d in range(D):
        n = anchor_targets[d]
        if not has_score[:, d].any():
            terms.append(F.term("anchor_bias", "absent", dimension=dim_ids[d], count=n,
                                reason="no readings on this dimension: a bias does not exist"))
            anchor_rows.append({"dimension": dim_ids[d], "state": "relative", "n_targets": n})
        else:
            state = "prior_only" if n < need else ("weak" if n < C.WEAK_ANCHOR_BELOW else "identified")
            terms.append(F.term("anchor_bias", state, dimension=dim_ids[d], count=n))
            anchor_rows.append({"dimension": dim_ids[d], "n_targets": n,
                                "state": {"prior_only": "relative", "weak": "weak", "identified": "anchored"}[state]})
        if kinds[d] == "binary":
            status = sum(1 for r, who in judges_on[d].items()
                         if len(who) >= 2 or any(len(cells[(j, r, d)]) >= 2 for j in who if j in anchors))
            terms.append(F.term("binary_status", _state(status), dimension=dim_ids[d], count=status))
    for j in range(J):
        row = declared[j]
        scored_any = bool(has_score[j].any() or has_pair[j].any())
        # family effects
        if not model_judge[j]:
            terms.append(F.term("family", "absent", judge=ids[j], reason="not a model judge"))
        elif not family_on[j]:
            terms.append(F.term("family", "off", judge=ids[j], count=len(out_families[j])))
        else:
            terms.append(F.term("family", "identified" if family_ok[j] else "prior_only", judge=ids[j],
                                count=len(out_families[j])))
        reference = row["reference"]
        author = reference.get("author", {}).get("model", {}).get("family") if reference["shown"] else None
        if not reference["shown"]:
            terms.append(F.term("reference_family", "absent", judge=ids[j], reason="no reference is shown"))
        elif not model_judge[j]:
            terms.append(F.term("reference_family", "absent", judge=ids[j], reason="not a model judge"))
        elif author is None:
            terms.append(F.term("reference_family", "absent", judge=ids[j],
                                reason="the reference was not written by a model"))
        elif author == row["model"]["family"]:
            terms.append(F.term("reference_family", "confounded", judge=ids[j], count=len(out_families[j])))
        elif not family_on[j]:
            terms.append(F.term("reference_family", "off", judge=ids[j], count=len(out_families[j])))
        else:
            seen = bool(family_ok[j]) and author in out_families[j]
            terms.append(F.term("reference_family", "identified" if seen else "prior_only", judge=ids[j],
                                count=len(out_families[j])))
        # halo
        if not halo_on[j]:
            terms.append(F.term("halo_split", "absent", judge=ids[j],
                                reason="the judge reads one dimension per target" if scored_any
                                else "the judge has no readings"))
        else:
            wide = int(sum(1 for d in range(D) if tau_state[j, d] in F.MEASURED_STATES))
            count = int(repeats_by[j].max())
            if panel[j]:
                twin = any(_same_setting(row, other) and halo_on[k] for k, other in enumerate(declared))
                state = _state(count) if count or not twin else "weak"
            else:
                state = "prior_only" if wide < 2 else _state(count)
            terms.append(F.term("halo_split", state, judge=ids[j], count=count))
        # judge by input
        count = int(ji_inputs[j].max())
        if not ji_on[j].any():
            terms.append(F.term("judge_input", "off", judge=ids[j], count=count))
        else:
            state = "identified" if count >= JUDGE_INPUT_NEEDS else ("weak" if count >= 2 else "prior_only")
            terms.append(F.term("judge_input", state, judge=ids[j], count=count))
        # pairs
        mine = [p for p in pairs if p["j"] == j]
        if not mine:
            for name in ("position", "sigma_pair", "tie_width"):
                terms.append(F.term(name, "absent", judge=ids[j], reason="the judge has no pairs"))
            continue
        orders: dict[tuple, set] = {}
        calls_per: dict[tuple, int] = {}
        for p in mine:
            key = (p["d"], min(p["ra"], p["rb"]), max(p["ra"], p["rb"]))
            calls_per[key] = calls_per.get(key, 0) + 1
            if p["first"] is not None:
                orders.setdefault(key, set()).add(p["first"])
        both = sum(1 for found in orders.values() if len(found) == 2)
        balanced = min(sum(1 for p in mine if p["o"] > 0), sum(1 for p in mine if p["o"] < 0))
        count = max(both, balanced if balanced >= 2 else 0)
        terms.append(F.term("position", _state(count), judge=ids[j], count=count))
        repeated = sum(1 for n in calls_per.values() if n >= 2)
        ties = sum(1 for p in mine if p["v"] == 1)
        terms.append(F.term("sigma_pair", _state(repeated), judge=ids[j], count=repeated))
        terms.append(F.term("tie_width", _state(min(repeated, ties)), judge=ids[j], count=min(repeated, ties)))

    # ---- records for the fit file ----------------------------------------------------------------
    used_anchor = any(row["state"] != "relative" for row in anchor_rows)
    anchor = {"kind": "person" if anchors and used_anchor else "judges_mean",
              "judges": [ids[j] for j in anchors], "min_anchor_targets": need, "dimensions": anchor_rows}
    judge_counts: dict[tuple, dict] = {}
    cell_calls: dict[tuple, dict] = {}
    unit_values: dict[tuple, list] = {}
    for (j, r, d), rows in cells.items():
        tally = judge_counts.setdefault((ids[j], dim_ids[d]), {"n_targets": 0, "n_calls": 0, "n_pairs": 0})
        tally["n_targets"] += 1
        tally["n_calls"] += len(rows)
        cell_calls.setdefault((runs[r]["run"], dim_ids[d]), {})[ids[j]] = len(rows)
        if not probit[d]:
            unit_values.setdefault((r, d), []).append((j, float(np.mean([readings[n]["y"] for n in rows])),
                                                       len(rows)))
    for p in pairs:
        tally = judge_counts.setdefault((ids[p["j"]], dim_ids[p["d"]]), {"n_targets": 0, "n_calls": 0, "n_pairs": 0})
        tally["n_pairs"] += 1
    counts = {"readings": len(study.scores), "readings_used": n_reading_rows, "pairs": len(study.pairs),
              "pairs_used": len(pairs)}
    options = {"structure": form[0] if form[0] != "factor" else f"factor:{form[1]}", "halo": halo, "cuts": cuts,
               "terms": switches, "components": None if components is None else list(components)}
    masks = {"has_score": has_score, "has_pair": has_pair, "halo_on": halo_on, "panel": panel,
             "model_judge": model_judge, "family_on": family_on, "ji_on": ji_on, "pairs_only": pairs_only,
             "probit": probit, "sigma_fixed": sigma_fixed}
    return Design(rubric=rubric, judges=judges, runs=runs, components=None if components is None else list(components),
                  index=index, dims=dims, options=options, data=data, terms=terms, anchor=anchor, notes=notes,
                  counts=counts, judge_counts=judge_counts, relative=[dim_ids[d] for d in range(D) if pairs_only[d]],
                  cell_calls=cell_calls, unit_values=unit_values, masks=masks)


def report(design: Design) -> dict:
    """The identification report as one object: what `eval fit --check-only --json` prints."""
    return {"counts": {"runs": design.data["R"], "inputs": design.data["I"], "arms": design.data["A"],
                       "dimensions": design.data["D"], "judges": design.data["J"], **design.counts,
                       "not_used": len(design.notes)},
            "options": design.options, "anchor": design.anchor, "terms": design.terms,
            "relative": design.relative, "notes": design.notes}


def report_lines(design: Design) -> list[str]:
    """The identification report as text: one line per term that exists, with the count behind it."""
    c = design.counts
    lines = [f"design: {design.data['R']} runs of {design.data['A']} arms on {design.data['I']} inputs, "
             f"{design.data['D']} dimensions, {design.data['J']} judges",
             f"readings: {c['readings_used']} of {c['readings']} used; pairs: {c['pairs_used']} of {c['pairs']} used; "
             f"{len(design.notes)} rows not used",
             f"anchor: {design.anchor['kind']}"]
    for row in design.anchor["dimensions"]:
        lines.append(f"  {row['dimension']}: {row['state']} on {row['n_targets']} anchored targets")
    lines.append("terms (identified, weak, prior_only, confounded, off):")
    for row in design.terms:
        if row["state"] == "absent":
            continue
        where = " ".join(x for x in (row["judge"], row["dimension"]) if x)
        text = f"  {row['term']:<16} {where:<40} {row['state']:<11} count {row['count']}"
        if row["state"] in ("prior_only", "weak"):
            text += f"  (needs: {row['needs']})"
        lines.append(text.rstrip())
    if design.relative:
        lines.append("relative (pairs only, no gate can use it): " + ", ".join(design.relative))
    return lines
