"""The `loopmath eval` command group: every command and its options, one handler per command.

`add(sub)` registers the group on the top-level parser. Each command calls `run(args)` in its module:

    eval scores import | check | add | export    scores.run
    eval simulate                                 simulate.run
    eval fit                                      fit.run
    eval judges                                   judges.run
    eval rubric check                             rubric.run
    eval value                                    value.run
    eval steps                                    steps.run
    eval labels next                              labels.run

`args.eval_command` holds the command's words after `eval` (`"scores import"`, `"fit"`), so a module with
several commands can branch on it. A module is imported only when its command runs.

Conventions for `run(args)`:

- Return an exit code from `loopmath.output` (`EXIT_OK` 0, `EXIT_USER` 1, `EXIT_NOT_FOUND` 2).
- Let a `contracts.ContractError` rise: the wrapper here prints `error: <file>: <place>: <problem>; <fix>`
  on stderr and returns 1. Under `--json` the CLI then prints the usual `loopmath.error/1` object.
- With `args.json`, print exactly one object with `loopmath.output.emit_json("loopmath.eval-<command>/1", ...)`.
- Write nothing on a refusal: validate first, then write with `contracts.write_text_atomic`.

This module imports the standard library, `contracts` and `loopmath.output` only, so `loopmath --help`
never loads numpy for it.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from ..output import EXIT_NOT_IMPLEMENTED, EXIT_USER
from .contracts import ContractError

EXPERIMENTAL = "experimental: "
GROUP_HELP = EXPERIMENTAL + "read rubric scores from judges and people: judge noise, value, next labels"
MODULES = ("scores", "simulate", "fit", "judges", "rubric", "value", "steps", "labels")
COMMANDS = ("scores import", "scores check", "scores add", "scores export", "simulate", "fit", "judges",
            "rubric check", "value", "steps", "labels next")
STRUCTURE_HELP = ("covariance form: full, diag or factor:K (default: full up to 8 dimensions, factor:2 above)")


def not_built(args: argparse.Namespace) -> int:
    """What every stub's `run(args)` returns: one line on stderr and exit code 3."""
    print(f"loopmath eval {getattr(args, 'eval_command', '')}: not built yet".replace(" :", ":"), file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def _handler(module: str):
    target = f"loopmath.eval.{module}:run"

    def run(args: argparse.Namespace) -> int:
        try:
            return importlib.import_module(f"loopmath.eval.{module}").run(args)
        except ContractError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USER

    run.target = target  # type: ignore[attr-defined]  # read by cli_registry.targets
    return run


def _command(parent, name: str, module: str, words: str, help_text: str) -> argparse.ArgumentParser:
    p = parent.add_parser(name, help=help_text, description=help_text[0].upper() + help_text[1:] + ".")
    p.add_argument("--json", action="store_true", help="print one JSON object on stdout")
    p.set_defaults(func=_handler(module), eval_command=words)
    return p


def _fit_reader(p: argparse.ArgumentParser, *, study: bool = True) -> None:
    """Options shared by the stages that read a fit file."""
    p.add_argument("fit", metavar="FIT", help="the fit file, fit.json (its draws file sits beside it)")
    p.add_argument("--allow-unconverged", action="store_true", help="read a fit that did not converge")
    if study:
        p.add_argument("--study", default=None, metavar="DIR",
                       help="the study folder the fit was made from (default: the folder of FIT)")
        p.add_argument("--strict", action="store_true",
                       help="refuse when the study files changed since the fit read them (default: warn)")


def add(sub) -> None:
    """Add `eval` and its commands to the top-level subparsers."""
    group = sub.add_parser("eval", help=GROUP_HELP,
                           description="Experimental: turn rubric scores from LLM judges and human reviewers into "
                                       "a quality estimate per configuration, a profile of every judge, the value "
                                       "of each configuration and the next labels a reviewer should give. "
                                       "loopmath runs no model and no judge here: it reads the scores you record.")
    es = group.add_subparsers(dest="eval_group", required=True, metavar="COMMAND")

    # eval scores
    scores = es.add_parser("scores", help="import, check, add and export score readings and pairs")
    ss = scores.add_subparsers(dest="eval_verb", required=True, metavar="VERB")
    q = _command(ss, "import", "scores", "scores import", "build or extend a study folder from recorded scores")
    q.add_argument("--ocp", nargs="+", default=None, metavar="FILE",
                   help="OCP run documents, with their score signals and preferences")
    q.add_argument("--rows", default=None, metavar="FILE", help="harness-runs rows with outcome.judges")
    q.add_argument("--csv", default=None, metavar="FILE",
                   help="readings: run, step, component, judge, call, value, item, of, observed_at")
    q.add_argument("--pairs-csv", default=None, metavar="FILE",
                   help="pairs: a, b, step, component, judge, call, shown_first, verdict, observed_at")
    q.add_argument("--items", default=None, metavar="FILE", help="an items file to add to the study")
    q.add_argument("--labels", default=None, metavar="PLAN",
                   help="the labels plan a filled form (--csv, --pairs-csv) answers: joins each row's assignment")
    q.add_argument("--rubric", default=None, metavar="FILE", help="the rubric the readings were scored under")
    q.add_argument("--pair-component", default=None, metavar="ID",
                   help="the component of a preference that names none (default: the overall component)")
    q.add_argument("--allow-simulated", action="store_true", help="take simulated runs")
    q.add_argument("--out", required=True, metavar="DIR", help="the study folder to build or extend")

    q = _command(ss, "check", "scores", "scores check", "validate a study folder against the rubric and the judges")
    q.add_argument("study", metavar="STUDY", help="the study folder")
    q.add_argument("--rubric", required=True, metavar="FILE")
    q.add_argument("--judges", default=None, metavar="FILE")
    q.add_argument("--require-assignment", action="store_true",
                   help="refuse a person's reading that names no labels assignment")

    q = _command(ss, "add", "scores", "scores add", "append one reading or one pair to a study folder")
    q.add_argument("study", metavar="STUDY", help="the study folder")
    q.add_argument("--rubric", required=True, metavar="FILE", help="the rubric the reading was scored under")
    q.add_argument("--run", default=None, metavar="RUN", help="the run scored (a reading)")
    q.add_argument("--pair", nargs=2, default=None, metavar=("RUN_A", "RUN_B"), help="the two runs compared (a pair)")
    q.add_argument("--step", default="final", metavar="STEP", help="the step whose output was scored (default final)")
    q.add_argument("--component", default=None, metavar="ID",
                   help="the rubric component (a pair defaults to the overall component)")
    q.add_argument("--judge", required=True, metavar="ID")
    q.add_argument("--value", default=None, metavar="V", help="the raw score on the component's scale")
    q.add_argument("--item", default=None, metavar="ID", help="the check item read (value 0 or 1)")
    q.add_argument("--of", type=int, default=None, metavar="N", help="read as a count: value items passed out of N")
    q.add_argument("--verdict", choices=["a", "b", "tie"], default=None, help="the verdict of a pair")
    q.add_argument("--shown-first", choices=["a", "b"], default=None, help="which output the judge saw first")
    q.add_argument("--call", default=None, metavar="ID", help="the judge call (default: a new call)")
    q.add_argument("--observed-at", default=None, metavar="TIME", help="ISO 8601 with offset (default: now)")
    q.add_argument("--replaces", default=None, metavar="ID", help="the reading or pair this one corrects")
    q.add_argument("--assignment", default=None, metavar="ID", help="the labels assignment that asked for it")
    q.add_argument("--id", default=None, metavar="ID", help="the id of the new reading or pair (default: derived)")

    q = _command(ss, "export", "scores", "scores export",
                 "print the OCP signals and preferences of a study as JSON Lines")
    q.add_argument("study", metavar="STUDY", help="the study folder")
    q.add_argument("--run", action="append", default=None, metavar="RUN", help="only this run (repeatable)")
    q.add_argument("--judges", nargs="+", action="extend", default=None, metavar="ID",
                   help="only the readings and pairs of these judges")

    # eval simulate
    q = _command(es, "simulate", "simulate", "simulate", "write a synthetic study with known truth")
    q.add_argument("spec", nargs="?", default=None, metavar="SPEC", help="a spec file, or a catalog name")
    q.add_argument("--list", action="store_true", help="list the catalog specs")
    q.add_argument("--out", default=None, metavar="DIR", help="the study folder to write")
    q.add_argument("--seed", type=int, default=0)

    # eval fit
    q = _command(es, "fit", "fit", "fit", "fit the quality and judge model to a study (needs the [bayes] extra)")
    q.add_argument("study", metavar="STUDY", help="the study folder")
    q.add_argument("--rubric", required=True, metavar="FILE")
    q.add_argument("--judges", required=True, metavar="FILE")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--structure", default=None, metavar="FORM", help=STRUCTURE_HELP)
    q.add_argument("--halo", choices=["factor", "full"], default="factor",
                   help="halo form (full only when every fitted scale is linear or fraction)")
    q.add_argument("--cuts", choices=["fixed", "free"], default="fixed",
                   help="cut points of levels scales (free: inner cut points of scales with 4 or more levels)")
    q.add_argument("--terms", action="append", default=None, metavar="NAME=on|off|auto",
                   help="switch an optional term (family, judge_input) on, off or auto (repeatable)")
    q.add_argument("--draws", type=int, default=None, metavar="N", help="posterior draws per chain")
    q.add_argument("--chains", type=int, default=None, metavar="N")
    q.add_argument("--keep-draws", type=int, default=1000, metavar="N",
                   help="draws kept in the draws file after thinning (default 1000)")
    q.add_argument("--components", nargs="+", default=None, metavar="ID", help="fit only these components")
    q.add_argument("--check-only", action="store_true", help="print the identification report and stop")
    q.add_argument("--allow-simulated", action="store_true", help="fit a study that holds simulated runs")
    q.add_argument("--out", default=None, metavar="FILE", help="the fit file to write (default: STUDY/fit.json)")
    q.add_argument("--engine", choices=["pymc", "laplace"], default="laplace",
                   help="laplace fits fast, with approximate draws, and needs no extra (the default); pymc samples")

    # eval judges
    q = _command(es, "judges", "judges", "judges", "each judge's bias, disagreement, noise and cost, from a fit")
    _fit_reader(q)
    q.add_argument("--target-error", type=float, default=None, metavar="X",
                   help="the error to reach: lists the cheapest judge and repeats that reach it")
    q.add_argument("--budget-usd", type=float, default=None, metavar="USD",
                   help="the budget per output: lists the best judge and repeats within it")

    # eval rubric check
    rubric = es.add_parser("rubric", help="validate a rubric and the files that name its components")
    rs = rubric.add_subparsers(dest="eval_verb", required=True, metavar="VERB")
    q = _command(rs, "check", "rubric", "rubric check",
                 "validate a rubric, and with --fit report its health from the data")
    q.add_argument("rubric", metavar="RUBRIC", help="the rubric file")
    q.add_argument("--judges", default=None, metavar="FILE")
    q.add_argument("--value", default=None, metavar="FILE")
    q.add_argument("--items", default=None, metavar="FILE")
    q.add_argument("--fit", default=None, metavar="FILE", help="add the rubric health report from this fit")
    q.add_argument("--allow-unconverged", action="store_true", help="read a fit that did not converge")

    # eval value
    q = _command(es, "value", "value", "value", "value and gate pass probability per configuration, from a fit")
    _fit_reader(q)
    q.add_argument("--value", required=True, metavar="FILE", help="the value file: weights, gates, objective")
    q.add_argument("--reference", default=None, metavar="ARM",
                   help="the arm others are compared against (default: the value file's objective.reference)")
    q.add_argument("--check", choices=["design"], default=None,
                   help="design: a design-based estimate of the mean human score beside the model's")
    q.add_argument("--labels", nargs="+", default=None, metavar="PLAN",
                   help="the labels plans whose inclusion probabilities --check design uses")
    q.add_argument("--seed", type=int, default=0)

    # eval steps
    q = _command(es, "steps", "steps", "steps", "which step drives the final value: measured swaps, hints, swaps not run")
    _fit_reader(q)
    q.add_argument("--value", required=True, metavar="FILE", help="the value file: weights, gates, objective")
    q.add_argument("--reference", default=None, metavar="ARM", help="the arm swaps are measured against")
    q.add_argument("--pool", action="store_true", help="add a main-effects fit across all arms (model-based)")
    q.add_argument("--runs", default=None, metavar="FILE", help="the runs file (default: runs.jsonl of the study)")
    q.add_argument("--seed", type=int, default=0)

    # eval labels next
    labels = es.add_parser("labels", help="what a human reviewer should score or compare next")
    ls = labels.add_subparsers(dest="eval_verb", required=True, metavar="VERB")
    q = _command(ls, "next", "labels", "labels next", "rank what a human reviewer should score or compare next")
    _fit_reader(q)
    q.add_argument("--value", default=None, metavar="FILE", help="the value file (active mode ranks by decision loss)")
    q.add_argument("--mode", choices=["pilot", "active"], default="pilot",
                   help="pilot: a blind stratified random sample; active: ranked by what a label teaches")
    q.add_argument("--n", type=int, default=None, metavar="N", help="the budget, in assignments")
    q.add_argument("--minutes", type=float, default=None, metavar="M", help="the budget, in reviewer minutes")
    q.add_argument("--ask", choices=["scores", "pairs", "auto"], default="scores")
    q.add_argument("--reviewer", default=None, metavar="ID", help="the person judge the assignments are for")
    q.add_argument("--overlap", default=None, metavar="ID", help="a second reviewer, for overlap assignments")
    q.add_argument("--components", nargs="+", default=None, metavar="ID", help="only these components")
    q.add_argument("--arms", nargs="+", default=None, metavar="ARM", help="only targets of these arms")
    q.add_argument("--uniform-share", type=float, default=0.2, metavar="X",
                   help="the share of the budget drawn uniformly at random (default 0.2)")
    q.add_argument("--repeat-share", type=float, default=0.1, metavar="X",
                   help="the share of assignments that repeat or overlap earlier ones (default 0.1)")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", default=None, metavar="FILE", help="the labels plan to write")
    q.add_argument("--form", default=None, metavar="FILE", help="also write a blank CSV for the reviewer")
    q.add_argument("--explain", default=None, metavar="ID", help="print why one assignment was ranked where it was")
