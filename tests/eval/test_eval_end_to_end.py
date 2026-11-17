"""`loopmath eval` end to end: the walk of the guide on every catalog spec.

The walk, with `eval simulate` in place of the user's own runs and judges:

    eval simulate SPEC --out study/
    eval rubric check rubric.json --judges judges.json [--value value.json] [--items items.jsonl]
    eval scores check study/ --rubric rubric.json --judges judges.json
    eval fit study/ ...                      then  eval judges study/fit.json
    eval labels next ... --mode pilot --form labels-001.csv
    (a reviewer fills the form)              then  eval scores import --csv ... --labels ...
    eval fit study/ ...                      again, with the reviewer's new readings
    eval value, eval steps, eval labels next --mode active

Two versions:

- The default one runs every stage of the walk but the sampler, on every catalog spec, in a few
  seconds. Where the walk fits, it writes the fit file from the simulated truth with
  `testing.build_fit`, so each later stage reads a fit of that spec's own design. A separate test
  runs `eval fit --check-only` (the identification report, no sampling) on every spec.
- The full one, marked `slow`, runs the walk with `eval fit` itself, twice per spec. It runs only
  with LOOPMATH_EVAL_SLOW=1. LOOPMATH_EVAL_RECOVERY_DRAWS and LOOPMATH_EVAL_RECOVERY_INPUTS make it
  smaller; LOOPMATH_EVAL_RECOVERY_REPORT names a folder for its numbers. A reduced run reads a fit
  that did not converge with `--allow-unconverged` and says so in its numbers; a run at the
  command's own draws does not, as in the guide.

A test that needs `eval fit` skips with "not built" while that command is a stub, and with the
install line when the optional modelling extra is missing. Every study here is simulated.
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import os
import shlex
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import simulate, specs, testing

SMALL = Path(__file__).parent / "data" / "study-small"
REPO = Path(__file__).parents[2]
slow = pytest.mark.slow
SHORT_INPUTS = 12          # inputs per study in the default version (the catalog's own size is 40)
PILOT_N = 8                # assignments asked of the reviewer in the pilot batch
STUDY_FILES = {"runs": C.RUNS_FILE, "scores": "scores.jsonl", "pairs": "pairs.jsonl", "items": "items.jsonl",
               "rubric": "rubric.json", "judges": "judges.json"}
FORM_COLUMNS = ["assignment", "run", "step", "component", "judge", "item", "of", "value", "observed_at"]


def env_int(name: str) -> int | None:
    """A whole number from the environment, or None when it is not set."""
    return int(os.environ[name]) if os.environ.get(name) else None


DRAWS = env_int("LOOPMATH_EVAL_RECOVERY_DRAWS")      # draws per chain of a slow fit (default: the command's own)
INPUTS = env_int("LOOPMATH_EVAL_RECOVERY_INPUTS")    # inputs per slow study (default: the catalog's own)


# ---------------------------------------------------------------------------------------------------
# The commands


def lm(*argv: object) -> tuple[int, str, str]:
    """`loopmath ARGV` in this process: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([str(word) for word in argv])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def js(*argv: object) -> dict:
    """The one JSON object a command prints under `--json`. The command must exit 0."""
    code, out, err = lm(*argv, "--json")
    assert code == 0, (argv, code, err.strip()[-600:])
    return json.loads(out)


@functools.lru_cache(maxsize=None)
def fit_state() -> str:
    """`built` when `eval fit` can sample here, else why not: `not built`, or the engine's install line."""
    code, _, err = lm("eval", "fit", SMALL, "--rubric", SMALL / "rubric.json", "--judges", SMALL / "judges.json",
                      "--check-only")
    if code == 3:
        return "not built"
    assert code == 0, err
    from loopmath.eval import engine_pymc

    if not engine_pymc.available():
        return "the engine is not installed: " + engine_pymc.INSTALL
    return "built"


def need_fit(sampling: bool = True) -> None:
    """Skip the calling test while `eval fit` is not built (or, for a sampling test, cannot sample)."""
    state = fit_state()
    if state == "not built" or (sampling and state != "built"):
        pytest.skip(f"eval fit: {state}")


# ---------------------------------------------------------------------------------------------------
# A simulated study and its reviewer


def simulate_study(name: str, folder: Path, seed: int = 0, inputs: int | None = None) -> dict:
    """`eval simulate` of a catalog spec into `folder`; with `inputs`, the same design on that many inputs."""
    spec: object = name
    if inputs is not None:
        spec = folder.parent / f"{folder.name}.spec.json"
        Path(spec).write_text(json.dumps(specs.get(name, inputs=inputs)), encoding="utf-8")
    got = js("eval", "simulate", spec, "--out", folder, "--seed", seed)
    assert got["simulated"] is True and got["counts"]["runs"] > 0
    return got


def value_file(folder: Path) -> Path:
    """The study's value file. A spec without one gets equal weights on the final criteria that have scores."""
    path = folder / "value.json"
    if path.exists():
        return path
    rubric = C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC)
    relative = set(json.loads((folder / simulate.TRUTH_FILE).read_text(encoding="utf-8")).get("relative") or ())
    weights = {dim["component"]: 1.0 for dim in C.dimensions(rubric)
               if dim["step"] == "final" and dim["role"] == "criterion" and dim["id"] not in relative}
    assert weights, "the spec has no scored criterion on the final step"
    path = folder.parent / f"{folder.name}.value.json"
    path.write_text(json.dumps({"schema": C.SCHEMA_VALUE, "weights": weights, "gates": [],
                                "objective": {"kind": "best_value"}}), encoding="utf-8")
    return path


def reviewer_value(dim: dict, m: float, rng: np.random.Generator, item: bool) -> object:
    """What a reviewer writes for a target whose reported quality on this dimension is `m`."""
    kind, lower = dim["scale"]["kind"], dim.get("better", "higher") == "lower"
    if kind == "binary":
        verdict = int(m >= 0.5)
        return 1 - verdict if lower else verdict
    if kind == "items":
        passed = int(rng.random() < m)
        assert item, "an items row of a form names its item"
        return 1 - passed if lower else passed
    own = float(Q.own_scale(min(1.0, max(0.0, m)), dim))
    if kind == "levels":
        names = dim["scale"]["levels"]
        return names[min(len(names) - 1, max(0, int(round(own)) - 1))]
    return round(own, 3)


def fill_form(form: Path, folder: Path, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Fill a blank readings form as a reviewer who sees the true reported quality of each output.

    `noise` is the sd of the reviewer's own error on a `linear` or `fraction` reading, on the 0 to 1
    scale. The filled rows are written back to `form` and returned."""
    truth = json.loads((folder / simulate.TRUTH_FILE).read_text(encoding="utf-8"))
    rubric = C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC)
    dims = C.dimensions(rubric)
    place = {dim["id"]: d for d, dim in enumerate(dims)}
    runs = {run: r for r, run in enumerate(truth["index"]["runs"])}
    reported = truth["reported"]["runs"]
    rng = np.random.default_rng(seed)
    with form.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FORM_COLUMNS, reader.fieldnames
        rows = list(reader)
    for row in rows:
        assert row["value"] == "", "the form comes blank"
        d = place[C.dimension_id(row["step"], row["component"])]
        m = float(reported[runs[row["run"]]][d])
        if noise and dims[d]["scale"]["kind"] in Q.UNIT_KINDS and dims[d]["scale"]["kind"] != "levels":
            m = m + noise * float(rng.standard_normal())
        row["value"] = reviewer_value(dims[d], m, rng, bool(row["item"]))
    with form.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------------------------------
# The two ways to get a fit file


def fit_from_truth(folder: Path, seed: int = 0) -> dict:
    """Write `fit.json` from the simulated truth, with no sampling: every draw is the truth plus a little noise.

    A parameter the design gives no reading for is null in the truth; `build_fit` needs a number there,
    so 0 stands in. The stages only have to run on this fit, not to report anything true from it."""
    truth = simulate.read_truth(folder / simulate.TRUTH_FILE)
    arrays = {}
    for name in F.DRAW_SHAPES:
        if name == "cuts" or truth.get(name) is None:
            continue
        value = np.asarray(truth[name], dtype=float)
        arrays[name] = np.nan_to_num(value, nan=0.0) if name in ("bias", "tau", "sigma") else value
    files = {role: folder / file for role, file in STUDY_FILES.items() if (folder / file).exists()}
    built = testing.build_fit(folder / "fit.json", rubric=C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC),
                              judges=C.read_document(folder / "judges.json", C.SCHEMA_JUDGES),
                              runs=C.read_study(folder).runs, truth=arrays, seed=seed,
                              relative=truth.get("relative") or (), inputs=C.input_digests(files))
    return {"converged": True, "seconds": 0.0, "flags": [], "fit": built.fit}


def fit_by_sampling(folder: Path, seed: int = 0, draws: int | None = DRAWS) -> dict:
    """`eval fit` on the study, on 2 chains. Returns the diagnostics, and the flag a later stage needs."""
    argv = ["eval", "fit", folder, "--rubric", folder / "rubric.json", "--judges", folder / "judges.json",
            "--seed", seed, "--out", folder / "fit.json", "--allow-simulated", "--chains", 2, "--engine", "pymc"]
    if draws is not None:
        argv += ["--draws", draws]
    start = time.monotonic()
    code, _, err = lm(*argv)
    assert code == 0, err.strip()[-800:]
    meta = json.loads((folder / "fit.json").read_text(encoding="utf-8"))
    converged = bool(meta["diagnostics"]["converged"])
    if draws is None:
        assert converged, ("the fit did not converge at the command's own draws", meta["diagnostics"])
    return {"converged": converged, "seconds": time.monotonic() - start, "diagnostics": meta["diagnostics"],
            "flags": [] if converged else ["--allow-unconverged"]}


# ---------------------------------------------------------------------------------------------------
# The walk


def walk(name: str, root: Path, fit, seed: int = 0, inputs: int | None = None) -> dict:
    """The walk of `docs/eval.md` on one catalog spec. `fit(folder, seed)` is the fit step. Returns what it counted."""
    folder = root / "study"
    simulated = simulate_study(name, folder, seed, inputs)
    rubric, judges = folder / "rubric.json", folder / "judges.json"
    value = value_file(folder)
    n_judges = len(json.loads(judges.read_text(encoding="utf-8"))["judges"])

    # 0. The user's files against each other.
    argv = ["eval", "rubric", "check", rubric, "--judges", judges, "--value", value]
    if (folder / "items.jsonl").exists():
        argv += ["--items", folder / "items.jsonl"]
    assert js(*argv)["ok"] is True

    # 1. The study against the rubric and the judges.
    checked = js("eval", "scores", "check", folder, "--rubric", rubric, "--judges", judges)
    assert checked["ok"] is True and checked["problems"] == [], checked["problems"]
    assert checked["counts"]["runs"] == simulated["counts"]["runs"]

    # 2. Fit, and read the judges.
    first = fit(folder, seed)
    flags = first["flags"]
    fit_json = folder / "fit.json"
    report = js("eval", "judges", fit_json, *flags)
    assert report["schema"] == "loopmath.eval-judges-report/1" and len(report["judges"]) == n_judges

    # 3. A blind random pilot for the reviewer, then their scores come back.
    plan_path, form = folder / "labels-001.json", root / "labels-001.csv"
    js("eval", "labels", "next", fit_json, "--mode", "pilot", "--n", PILOT_N, "--seed", 0, "--out", plan_path,
       "--form", form, *flags)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == C.SCHEMA_LABELS and plan["assignments"]
    for assignment in plan["assignments"]:
        assert not {"arm", "model", "family", "judge_scores"} & set(assignment), assignment
    rows = fill_form(form, folder, seed)
    assert rows and {row["assignment"] for row in rows} <= {a["id"] for a in plan["assignments"]}
    imported = js("eval", "scores", "import", "--csv", form, "--labels", plan_path, "--rubric", rubric,
                  "--out", folder)
    assert imported["readings"] == {"new": len(rows), "unchanged": 0, "refused": 0}, imported
    again = js("eval", "scores", "import", "--csv", form, "--labels", plan_path, "--rubric", rubric, "--out", folder)
    assert again["readings"] == {"new": 0, "unchanged": len(rows), "refused": 0} and again["written"] == []
    rechecked = js("eval", "scores", "check", folder, "--rubric", rubric, "--judges", judges)
    assert rechecked["ok"] is True
    assert rechecked["counts"]["readings"] == checked["counts"]["readings"] + len(rows)
    asked = {a["id"] for a in plan["assignments"]}
    labelled = [row for row in C.read_study(folder).scores if row.get("assignment") in asked]
    assert len(labelled) == len(rows)

    # 4. Refit, now with the reviewer's new readings.
    second = fit(folder, seed)
    flags = second["flags"]

    # 5. Value and gates, the step report, and the next, targeted batch of labels.
    arms = json.loads(fit_json.read_text(encoding="utf-8"))["index"]["arms"]
    valued = js("eval", "value", fit_json, "--value", value, *flags)
    assert [arm["arm"] for arm in valued["arms"]] == arms
    assert len(valued["comparisons"]) == len(arms) - 1
    for arm in valued["arms"]:
        assert arm["value"]["lo"] <= arm["value"]["mean"] <= arm["value"]["hi"]
        assert 0.0 <= arm["p_pass"]["mean"] <= 1.0
    steps = js("eval", "steps", fit_json, "--value", value, "--reference", arms[0], *flags)
    assert steps["reference"] == arms[0]
    assert all(isinstance(steps[part], list) for part in ("measured", "hint", "not_run"))
    assert all(entry["basis"].startswith("measured") for entry in steps["measured"])
    assert all(entry["basis"].startswith("correlational") for entry in steps["hint"])
    active_path = folder / "labels-002.json"
    js("eval", "labels", "next", fit_json, "--value", value, "--mode", "active", "--minutes", 60, "--seed", 1,
       "--out", active_path, *flags)
    active = json.loads(active_path.read_text(encoding="utf-8"))
    assert active["schema"] == C.SCHEMA_LABELS and active["assignments"]
    assert sum(a["minutes"] for a in active["assignments"]) <= 60 * 1.25
    for assignment in active["assignments"]:
        assert not {"arm", "model", "family", "judge_scores"} & set(assignment), assignment
        assert 0.0 < assignment["inclusion_probability"] <= 1.0
    return {"spec": name, "seed": seed, "counts": simulated["counts"], "pilot_rows": len(rows),
            "active_assignments": len(active["assignments"]), "measured_swaps": len(steps["measured"]),
            "hints": len(steps["hint"]), "swaps_not_run": len(steps["not_run"]),
            "fits": [{key: one[key] for key in ("converged", "seconds", "diagnostics") if key in one}
                     for one in (first, second)]}


def write_report(name: str, payload: dict) -> None:
    """Keep a slow test's numbers in the folder LOOPMATH_EVAL_RECOVERY_REPORT names, when it is set."""
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# The tests


def real_parser() -> argparse.ArgumentParser:
    """The top-level parser, as `cli.main` builds it."""
    class Built(Exception):
        pass

    def grab(self, *args, **kwargs):
        raise Built(self)

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def written_commands(text: str) -> list[str]:
    """Every `loopmath eval ...` line of the code blocks of a document."""
    found, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside and line.lstrip("$ ").startswith("loopmath eval "):
            found.append(line.lstrip("$ ").strip())
    return found


@pytest.mark.parametrize("document, at_least", [("docs/eval.md", 12), ("README.md", 6)])
def test_every_eval_command_the_guide_and_the_readme_show_parses(document, at_least, capsys):
    """Each `loopmath eval` line shown in `docs/eval.md` and the README names a real command with real flags."""
    top = real_parser()
    lines = written_commands((REPO / document).read_text(encoding="utf-8"))
    assert len(lines) >= at_least, lines
    for line in lines:
        try:
            top.parse_args(shlex.split(line)[1:])
        except SystemExit:
            raise AssertionError(f"{document}: does not parse: {line}: {capsys.readouterr().err.strip()[-300:]}")
    if document == "docs/eval.md":      # the guide shows the whole walk
        for command in ("rubric check", "scores import", "scores check", "simulate", "fit", "judges",
                        "labels next", "value", "steps"):
            assert any(line.startswith(f"loopmath eval {command} ") for line in lines), command


def test_the_catalog_is_the_one_the_walk_covers():
    """Every spec the release names is in the catalog, so the parametrised tests below reach it."""
    named = {"one-component", "halo", "own-family", "reference-family", "gated", "cheap-vs-strong-judge",
             "levels-5", "levels-3", "binary-checks", "case-items", "pairs", "overall-vs-criteria",
             "panel-vs-per-component", "steps-swap", "mixed-rubric"}
    assert named <= set(specs.names())


@pytest.mark.parametrize("name", specs.names())
def test_the_walk_without_the_sampler_on_every_catalog_spec(name, tmp_path):
    """The default version: every stage but the sampler, on a fit file written from the simulated truth."""
    got = walk(name, tmp_path, fit_from_truth, inputs=SHORT_INPUTS)
    assert got["pilot_rows"] >= PILOT_N - 2 and got["active_assignments"] >= 1


@pytest.mark.parametrize("name", specs.names())
def test_the_identification_report_on_every_catalog_spec(name, tmp_path, capsys):
    """`eval fit --check-only` reads every catalog study, prints the report of what the design can
    identify, and writes nothing. Skips with "not built" while `eval fit` is a stub."""
    need_fit(sampling=False)
    folder = tmp_path / "study"
    simulate_study(name, folder, inputs=SHORT_INPUTS)
    before = sorted(path.name for path in folder.iterdir())
    code, out, err = lm("eval", "fit", folder, "--rubric", folder / "rubric.json", "--judges",
                        folder / "judges.json", "--check-only", "--allow-simulated")
    assert code == 0, err
    assert "terms" in out and "anchor" in out
    assert sorted(path.name for path in folder.iterdir()) == before
    refused, _, err = lm("eval", "fit", folder, "--rubric", folder / "rubric.json", "--judges",
                         folder / "judges.json", "--check-only")
    assert refused == 1 and "--allow-simulated" in err   # a simulated study is never fitted by accident


@slow
@pytest.mark.parametrize("name", specs.names())
def test_the_full_walk_with_eval_fit_on_every_catalog_spec(name, tmp_path):
    """The full version: the walk with `eval fit` itself, twice per spec (before and after the pilot labels)."""
    need_fit()
    got = walk(name, tmp_path, fit_by_sampling, inputs=INPUTS)
    got["draws"], got["inputs"] = DRAWS, INPUTS
    write_report(f"end-to-end-{name}", got)
    assert got["pilot_rows"] >= PILOT_N - 2 and got["active_assignments"] >= 1
