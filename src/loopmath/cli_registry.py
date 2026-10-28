"""The 0.1.0 command registry (design/0.1/02-commands.md).

Every command and flag of the spec is declared here, and each command points to a
handler `module:function` in the owning package. Handlers are imported only
when their command runs, so a package that fails to import never breaks `--help` or
another command. A new flag is declared here first.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .output import EXIT_OK, HTML_DEFAULT, emit_json


def _lazy(target: str):
    module_name, fn_name = target.split(":")

    def run(args: argparse.Namespace) -> int:
        module = importlib.import_module(module_name)
        return getattr(module, fn_name)(args)

    run.target = target  # type: ignore[attr-defined]
    return run


HOME_HELP = ("store root (default: $LOOPMATH_HOME or ~/.loopmath); the log parse cache follows it unless "
             "LOOPMATH_CACHE_DIR is set")


def _common(p: argparse.ArgumentParser, *, json_flag: bool = True, html: bool = False,
            home_help: str = HOME_HELP) -> None:
    p.add_argument("--home", default=None, metavar="PATH", help=home_help)
    if json_flag:
        p.add_argument("--json", action="store_true", help="print one JSON object on stdout")
    if html:
        p.add_argument("--html", nargs="?", const=HTML_DEFAULT, default=None, metavar="PATH",
                       help="write a self-contained HTML view (default path under $LOOPMATH_HOME/views/)")


HORIZON_HELP = ("the run's wall-clock time budget: 8h, 90m or seconds; none for open-ended "
                "(default: the one the fit recorded for this task, else open-ended)")


def _task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-file", default=None, metavar="FILE", help="the task as a JSON file, instead of the flags below")
    p.add_argument("--type", dest="task_type", default=None, metavar="T", help="task type (see loopmath task-types)")
    p.add_argument("--repo", default=None, metavar="R", help="repository as owner/name from its origin remote, e.g. acme/api (else its folder name)")
    p.add_argument("--subtype", default=None, metavar="S", help="your own subtype of the task type (config subtypes)")
    p.add_argument("--title", default=None, metavar="TEXT", help="a short title for the task")
    p.add_argument("--feature", action="append", default=[], metavar="K=V", help="task feature (repeatable)")
    p.add_argument("--horizon", default=None, metavar="TIME", help=HORIZON_HELP)
    p.add_argument("--base-commit", default=None, metavar="SHA", help="the commit the work starts from")


def _check_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--check", default=None, metavar="CHK", help="record into this check-change check: the run gets "
                   "its pair's task and slate (with --arm)")
    p.add_argument("--screen", default=None, metavar="SCR", help="experimental: record into this screen: the run "
                   "gets its pair's task and slate (with --arm A or B)")
    p.add_argument("--arm", default=None, choices=["base", "cand", "A", "B"], help="the check arm the run is (base "
                   "or cand, with --check), or the screen arm (A or B, with --screen)")
    p.add_argument("--pair", default=None, metavar="PNN", help="the check's or screen's pair (default: the earliest "
                   "one missing this arm)")
    p.add_argument("--label", action="append", default=[], metavar="KEY=VALUE", help="a run label (repeatable), "
                   "e.g. config=new; check-change from-runs picks runs by label")


def _add_compare(sub) -> None:
    p = sub.add_parser("check-change", help="paired A/B check of a change: start, status, from-runs",
                       description="A paired A/B check of a change. Start a check, record its pairs, and read its "
                                   "comparison and verdict.")
    cs = p.add_subparsers(dest="check_command", required=True, metavar="{start,status,from-runs}")

    def decision(q: argparse.ArgumentParser) -> None:
        q.add_argument("--rule", default=None, metavar="RULE", help="what counts as accepted: accept (default: each "
                       "run's own rule, as runs and recommend read it), tests, a named rule, review, NAME>=X, or "
                       "score:NAME[:lower] for a raw score")
        q.add_argument("--budget-usd", type=float, default=None, metavar="USD", help="dollars the check may spend")
        q.add_argument("--objective", default=None, choices=["cpar", "gain"], help="cost per accepted result "
                       "(default for an acceptance rule) or the gain in acceptance or score")
        q.add_argument("--floor", type=float, default=None, metavar="F", help="the smallest improvement that counts "
                       "(default 0.05; 0 for a score); a lower floor states no false adopt bound")
        q.add_argument("--confidence", type=float, default=None, metavar="C", help="reject at P(better) <= 1 - C; "
                       "adopt at the calibrated threshold for alpha = 1 - C (default 0.95)")
        q.add_argument("--horizon-runs", type=float, default=None, metavar="N", help="the number of future runs "
                       "the change will serve, which prices the next pair (default 50; for example 500 for a "
                       "long-lived change)")
        q.add_argument("--value-usd", type=float, default=None, metavar="USD", help="dollars per point of gain "
                       "(default: the base's cost per accepted result)")
        q.add_argument("--value-stop", action="store_true", default=None, help="stop the check, keeping the base, "
                       "only when the next pairs are certified not worth their price: every plan's net value below "
                       "0 by more than two standard errors, or no look left can change the action; an unresolved "
                       "value never stops. At price 0 it fires only when going on is certified to lose, or when no "
                       "look left can change the action; a sample with no adoption never fires. By default the "
                       "value and the price are shown at each look and the check goes on to a decision, its cap or "
                       "its budget. Use it when a pair is dear against what an adoption is worth over "
                       "--horizon-runs: in simulated checks (cap 24, an acceptance rule), at the default 50 future "
                       "runs it saved 6 to 16 pairs ($2.90 to $10.33) a check but dropped 26 to 40 percent of the "
                       "adoptions of real cuts; at 500 runs it saved 10 to 14 pairs ($6.22 to $9.02) a check and "
                       "dropped none")
        q.add_argument("--min-pairs", type=int, default=None, metavar="K", help="with --value-stop, the scored "
                       "pairs before the value may stop the check (default 4; 1 stops from the first pair)")
        q.add_argument("--pair-price", type=float, default=None, metavar="USD", help="the price of one pair that "
                       "the next pairs' value is compared with (default: its expected cost); 0 for runs on a "
                       "subscription, with no marginal dollars: the budget still counts the runs' recorded dollars")

    q = cs.add_parser("start", help="start a check: its pairs, each pair's arm order, and the next pair to run")
    q.add_argument("--base", required=True, metavar="LABEL|PATH", help="the base arm: a label, or its harness tree")
    q.add_argument("--cand", required=True, metavar="LABEL|PATH", help="the candidate arm: a label, or its harness tree")
    q.add_argument("--base-set", action="append", default=None, metavar="PIECE=HARNESS:MODEL:EFFORT",
                   help="the base arm's setting for a piece, as run record --set takes it (repeatable); its runs "
                   "take it, a run with another is refused, and status names any run whose log ran another")
    q.add_argument("--cand-set", action="append", default=None, metavar="PIECE=HARNESS:MODEL:EFFORT",
                   help="the candidate arm's setting, as --base-set: a model switch is two arms by --set")
    tasks = q.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks", default=None, metavar="FILE.jsonl", help="one task per line; one pair per task")
    tasks.add_argument("--n", type=int, default=None, metavar="N", help="N pairs, each taking its task from its "
                       "first run")
    decision(q)
    q.add_argument("--seed", type=int, default=0, metavar="S", help="seed of the arm orders and the comparison")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.compare.commands:check_start"))

    q = cs.add_parser("status", help="the check's comparison, verdict and next pair")
    q.add_argument("check", nargs="?", default=None, metavar="CHK", help="the check (default: the newest)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.compare.commands:check_status"))

    q = cs.add_parser("from-runs", help="compare finished runs already in the store, chosen by label")
    q.add_argument("--base-label", required=True, metavar="SEL", help="the base runs: KEY=VALUE (a run label), "
                   "CHK:base|cand, a cfg_ configuration id, a harness digest as sha256:<hex> or digest:sha256:<hex> "
                   "(a prefix of one digest will do), arm:NAME, or a word (label arm=WORD)")
    q.add_argument("--cand-label", required=True, metavar="SEL", help="the candidate runs, as --base-label")
    decision(q)
    q.add_argument("--max-pairs", type=int, default=None, metavar="N", help="the cap the false adopt bound is "
                   "stated for (default: the smallest calibrated cap above the pairs recorded, 6, 12, 24 or 48; "
                   "past 48 pairs, no bound is stated and adopting needs the strictest threshold)")
    q.add_argument("--unit", choices=("case", "run"), default=None, help="what is paired: case, one pair per "
                   "case (task) with each arm's mean score, or run (default: case under a score rule when a case "
                   "has more than one run on an arm, since runs of one case are not independent pairs for new "
                   "cases; else run; acceptance rows are compared run by run)")
    q.add_argument("--seed", type=int, default=0, metavar="S", help="seed of the comparison's draws")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.compare.commands:check_from_runs"))


SCREEN_DESCRIPTION = """Experimental: screen up to about 16 atomic harness edits in one design. Each pair
runs a task with some edits on (arm A) and with the others on (arm B); status
ranks the edits on success and cost and names the predicted-best subset. Its
ranges are against the base (every edit off), which no arm runs, so the effect
of having the edited files at all is not in them."""

SCREEN_FLOW = """the flow:
  loopmath screen start --base harness --edits edits --tasks tasks.jsonl
  loopmath screen next
  loopmath run record --screen SCR --arm A --since START --verified tests=pass
  loopmath screen status
  loopmath screen apply --best --out best-harness
  loopmath check-change start --base harness --cand best-harness --tasks other-tasks.jsonl
next prints a pair's arms, their trees and the exact lines that record each
run; run and record both arms, and repeat until every pair is recorded or
status says stop. Then check the best subset against the base on other tasks:
the screen's ranges guide the choice, the check decides it."""


def _add_screen(sub) -> None:
    p = sub.add_parser("screen", help="experimental: fold-over screen of harness edits: start, next, status, apply",
                       description=SCREEN_DESCRIPTION, epilog=SCREEN_FLOW,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    cs = p.add_subparsers(dest="screen_command", required=True, metavar="{start,next,status,apply}")

    q = cs.add_parser("start", help="experimental: start a screen: its pairs, each pair's fold-over row, and the next "
                      "pair", description="Experimental: start a screen: one pair per task, each pair a fold-over row "
                      "of edits on and off, and both arms' trees built from the base and the edits.")
    q.add_argument("--base", required=True, metavar="PATH", help="the base harness tree the edits change")
    q.add_argument("--edits", required=True, metavar="DIR", help="one subfolder per edit, laid over the base: "
                   "PATH copies, PATH.append appends, PATH.remove removes lines")
    tasks = q.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks", default=None, metavar="FILE.jsonl", help="one task per line; one pair per task")
    tasks.add_argument("--n", type=int, default=None, metavar="N", help="N pairs, each taking its task from its "
                       "first run")
    q.add_argument("--pairs", type=int, default=None, metavar="N", help="use the first N tasks of --tasks")
    q.add_argument("--rule", default=None, metavar="RULE", help="what counts as accepted, as in check-change "
                   "(default accept: each run's own rule)")
    q.add_argument("--objective", default=None, choices=["cpar", "cost", "gain"], help="what the best subset "
                   "maximises: the cut in cost per accepted result (default for an acceptance rule), in cost per "
                   "run, or the gain in success (default for a score rule)")
    q.add_argument("--confidence", type=float, default=0.9, metavar="C", help="an edit is settled at P(helps) of "
                   "C or 1 - C (default 0.9); the screen stops when every edit is")
    q.add_argument("--horizon-runs", type=float, default=50, metavar="N", help="future runs the best harness "
                   "serves, which prices the next pair (default 50)")
    q.add_argument("--min-pairs", type=int, default=4, metavar="K", help="scored pairs before the next pair's "
                   "value may stop the screen (default 4)")
    q.add_argument("--pair-usd", type=float, default=None, metavar="USD", help="the price of one more pair "
                   "(default: twice the mean run cost; 0 on an allowance, so only the cap or settling stops)")
    q.add_argument("--seed", type=int, default=0, metavar="S", help="seed of the design, the arm orders and the draws")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.screen.commands:screen_start"))

    q = cs.add_parser("next", help="experimental: the next pair: its task, each arm's edits and built tree, the "
                      "record lines", description="Experimental: the next pair to run: its task, each arm's edits "
                      "and built tree, and the lines that record its two runs.")
    q.add_argument("screen", nargs="?", default=None, metavar="SCR", help="the screen (default: the newest)")
    q.add_argument("--pair", default=None, metavar="PNN", help="this pair instead, for a driver with several pairs "
                   "in flight")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.screen.commands:screen_next"))

    q = cs.add_parser("status", help="experimental: the edits ranked with ranges, the predicted-best subset and the "
                      "next pair's worth", description="Experimental: the edits ranked on success and cost with 80% "
                      "ranges, the predicted-best subset against the base, and the next pair's worth. The ranges "
                      "assume the edits add up from the base, which no arm runs.")
    q.add_argument("screen", nargs="?", default=None, metavar="SCR", help="the screen (default: the newest)")
    q.add_argument("--on", default=None, metavar="EDIT,EDIT", help="also predict this subset of edits against the "
                   "base, run or not")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.screen.commands:screen_status"))

    q = cs.add_parser("apply", help="experimental: write the base with the best (or the given) edits to a new folder",
                      description="Experimental: write the base with the predicted-best (or the given) edits to a new "
                      "folder, ready for a check against the base.")
    q.add_argument("screen", nargs="?", default=None, metavar="SCR", help="the screen (default: the newest)")
    q.add_argument("--best", action="store_true", help="the predicted-best subset, as status names it")
    q.add_argument("--on", default=None, metavar="EDIT,EDIT", help="these edits instead")
    q.add_argument("--out", required=True, metavar="DIR", help="a new folder")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.screen.commands:screen_apply"))


def task_types_verb(args: argparse.Namespace) -> int:
    from .output import home
    from .store.config import Config, ConfigError
    from .taskmodel import task_types_payload

    try:  # the store's declared features, read without creating the store
        conf = Config.load(home(getattr(args, "home", None)) / "config.toml")
    except ConfigError as exc:
        print(f"warning: {exc}; showing the built-in features", file=sys.stderr)
        features, subtypes = None, None
    else:
        features, subtypes = conf.features_or_builtin(), conf.get("subtypes")
    payload = task_types_payload(subtypes, features)
    if args.json:
        emit_json(payload["schema"], payload)
        return EXIT_OK
    for t in payload["types"]:
        print(f"{t['id']:<10} {t['description']}")
    print()
    for f in payload["features"]:
        values = "|".join(f["values"]) or "free text"
        scope = f"({', '.join(f['types'])} only)" if f.get("types") else ""
        print((f"--feature {f['key']}=<{values}>  " + " ".join(x for x in (f["description"], scope) if x)).rstrip())
    print(f"--horizon <8h|90m|seconds|none>  {payload['horizon']['description']}")
    return EXIT_OK


FIT_HELP = "read this kept fit instead of the latest one: a fit_ id from `loopmath status` or the fits folder"


# ---------------------------------------------------------------- planning
def _add_planning(sub) -> None:
    p = sub.add_parser("task-types", help="list task types and task features")
    _common(p)
    p.set_defaults(func=task_types_verb)

    p = sub.add_parser("workflows", help="catalog and user workflows: list, show, validate, diff")
    ws = p.add_subparsers(dest="workflows_command", required=True)
    q = ws.add_parser("list", help="list catalog and user workflows")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.workflows.commands:list_workflows"))
    q = ws.add_parser("show", help="show one workflow")
    q.add_argument("id", metavar="ID", help="a catalog name, a user workflow (id or file stem), a file (workflow TOML, configuration or run JSON, run folder) or a cfg_ id")
    _common(q, html=True)
    q.set_defaults(func=_lazy("loopmath.workflows.commands:show"))
    q = ws.add_parser("validate", help="validate a workflow TOML file")
    q.add_argument("file", metavar="FILE", help="a workflow TOML file")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.workflows.commands:validate"))
    q = ws.add_parser("diff", help="differences between two workflows or configurations")
    q.add_argument("a", metavar="A", help="a catalog name, a user workflow (id or file stem), a file (workflow TOML, configuration or run JSON, run folder) or a cfg_ id")
    q.add_argument("b", metavar="B", help="the same kinds as A")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.workflows.commands:diff"))

    p = sub.add_parser("recommend", help="choices for one task (options 1 to 5) with chance and cost, the success-cost curve")
    _task_args(p)
    rule = p.add_mutually_exclusive_group()
    rule.add_argument("--rule", default=None, metavar="RULE", help="a named acceptance rule from config")
    rule.add_argument("--target", default=None, metavar="NAME>=X", help="score-target rule for this call, quoted: the shell reads > and < as redirections; e.g. 'heldout_perf>=2400' or 'runtime_s<=200'")
    p.add_argument("--goal", default=None, choices=["default", "p50", "p70", "p80", "p90", "p95", "p99"],
                   help="the pick to aim for: default (lowest expected dollars to an accepted result) or the curve row for that chance of success (default: config goal)")
    p.add_argument("--usual", default=None, metavar="CFG", help="a configuration id to treat as the usual one: "
                   "the [cfg_...] printed after a workflow")
    p.add_argument("--workflow", action="append", default=[], metavar="FILE.toml", help="also consider this workflow (repeatable)")
    p.add_argument("--models", default=None, metavar="M,M,...", help="restrict candidate models")
    p.add_argument("--fit", default=None, metavar="ID", help=FIT_HELP)
    p.add_argument("--brief", action="store_true", help="with --json: a short object for agents (task, reference, "
                   "goal, rescue, choices, message, rec, and page with --html); the stored recommendation keeps everything")
    p.add_argument("--harness-variants", nargs="?", const=5, type=int, default=None, metavar="N",
                   help="experimental: also price the N harness trees recorded on your runs that the fit saw most "
                   "(default 5), for the default pick and the goal, against the bare harness label on the same draws")
    p.add_argument("--harness", default=None, metavar="PATH|auto", help="experimental: also price this harness tree "
                   "as the first variant (auto: the git working tree of this folder), even when no run recorded it")
    p.add_argument("--role", default=None, metavar="ROLE", help="experimental: which model to give one role, such "
                   "as the mutator of an evolution loop: each model's chance that one step of that role improves, "
                   "its dollars per step and per improvement, on the runs recorded for that role (--models: these "
                   "models only, recorded or not)")
    p.add_argument("--unit", default=None, metavar="UNIT", help="with --role, when its runs come in more than one "
                   "unit: query, evaluated child or stored child")
    _common(p, html=True)
    p.set_defaults(func=_lazy("loopmath.recommend.commands:recommend"))

    # Hidden: no help= keeps it out of the command list, and HIDDEN keeps it out of the usage line.
    p = sub.add_parser("plan", description="Experimental, and hidden from loopmath --help: slates worth running "
                                           "over a backlog, within a budget, for designed experiments.")
    p.add_argument("--backlog", required=True, metavar="tasks.jsonl", help='one task as JSON per line, such as '
                   '{"type": "bug_fix", "repo": "acme/api", "title": "..."}')
    p.add_argument("--budget-usd", required=True, type=float, metavar="X", help="dollars to spend over the whole backlog")
    p.add_argument("--slate-size", type=int, default=2, help="runs per slate (default 2)")
    p.add_argument("--max-slates", type=int, default=None, metavar="N", help="stop after N slates")
    p.add_argument("--members", default="goal+explore", choices=["goal+explore", "grid"],
                   help="goal plus exploration picks from the fit, or a covering grid over the catalog when there is no fit")
    p.add_argument("--models", default=None, metavar="M,M,...", help="restrict candidate models")
    rule = p.add_mutually_exclusive_group()
    rule.add_argument("--rule", default=None, metavar="RULE", help="a named acceptance rule from config")
    rule.add_argument("--target", default=None, metavar="NAME>=X", help="score-target rule, quoted: e.g. 'heldout_perf>=2400'")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.recommend.commands:plan"))


# ---------------------------------------------------------------- recording
SOURCES = ["usual", "alternative", "exploration", "user_edit", "habit", "designed"]
TIER_CHOICES = ["verified", "reported", "heuristic", "asserted"]


def _add_recording(sub) -> None:
    p = sub.add_parser("run", help="record a run: start, record, import, attempt, artifact, finish, sessions")
    rs = p.add_subparsers(dest="run_command", required=True)

    q = rs.add_parser("start", help="open a run (OCP v0.3 skeleton in the store)")
    _task_args(q)
    q.add_argument("--config", default=None, metavar="CFG", help="configuration id from recommend")
    q.add_argument("--workflow", default=None, metavar="FILE.toml", help="a workflow TOML file, a catalog name or a workflow in "
                   "your store; its settings are used, and --set sets or overrides a piece")
    q.add_argument("--set", action="append", default=[], metavar="PIECE=HARNESS:MODEL:EFFORT", help="piece setting (repeatable)")
    q.add_argument("--source", default=None, choices=SOURCES, help="why this configuration was chosen (not with --choice)")
    q.add_argument("--rec", default=None, metavar="REC", help="the recommend id this run follows (stores a before-receipt)")
    q.add_argument("--choice", default=None, metavar="KEY", help="start a choice from recommend's choices (goal, pair, "
                   "reference, most_likely, cheapest_run) with --rec: the task, configuration and source come from it; a pair "
                   "opens a slate with both runs")
    slate = q.add_mutually_exclusive_group()
    slate.add_argument("--slate", default=None, metavar="SLT", help="add the run to this slate")
    slate.add_argument("--new-slate", action="store_true", help="open a new slate with this run in it")
    q.add_argument("--rule", default=None, metavar="RULE", help="a named acceptance rule from config; with --rec "
                   "it replaces the rule the recommendation was made for")
    _check_args(q)
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_start"))

    q = rs.add_parser("attempt", help="add or settle an attempt")
    q.add_argument("--run", required=True, metavar="RUN", help="run id from run start")
    q.add_argument("--end", default=None, metavar="ATT", help="settle this attempt instead of adding one")
    q.add_argument("--status", default=None, choices=["done", "failed", "rejected", "canceled", "lost"], help="how the attempt ended (with --end)")
    q.add_argument("--ended-at", default=None, metavar="TS", help="ISO time it ended (default: now)")
    q.add_argument("--piece", default=None, metavar="VERTEX", help="the workflow piece this attempt runs")
    q.add_argument("--harness", default=None, choices=["claude-code", "codex", "command"], help="what runs the attempt")
    q.add_argument("--model", default=None, metavar="M", help="model name")
    q.add_argument("--effort", default=None, metavar="E", help="reasoning effort")
    q.add_argument("--cwd", default=None, metavar="PATH", help="the folder the agent works in; with no --session its log is matched by this folder, a session that started in the attempt's window, and the model")
    q.add_argument("--session", default=None, metavar="ID", help="session id, or self for the orchestrator's own Claude Code session, counted only inside this attempt's started-at to ended-at window (outside Claude Code: the one Claude Code session in this folder active in the last 10 minutes; run sessions lists the ids)")
    q.add_argument("--round", type=int, default=1, metavar="K", help="round of this piece, 1 for the first try")
    q.add_argument("--cause", default="initial", choices=["initial", "sent_back", "gate_failed", "followup", "other"],
                   help="why this attempt was started")
    q.add_argument("--started-at", default=None, metavar="TS", help="ISO time it started (default: now)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_attempt"))

    q = rs.add_parser("artifact", help="record an artifact")
    q.add_argument("--run", required=True, metavar="RUN", help="run id from run start")
    q.add_argument("--kind", required=True, metavar="KIND", help="what the artifact is, e.g. plan, diff, review")
    q.add_argument("--path", required=True, metavar="P", help="where it is")
    q.add_argument("--by", required=True, metavar="ATT", help="the attempt that wrote it")
    q.add_argument("--read-by", action="append", default=[], metavar="ATT", help="an attempt that read it (repeatable)")
    q.add_argument("--supersedes", default=None, metavar="ART", help="the artifact this one replaces")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_artifact"))

    q = rs.add_parser("finish", help="match logs, cost, validate, receipt, background refit")
    q.add_argument("--run", required=True, metavar="RUN", help="run id from run start")
    q.add_argument("--harness", default=None, metavar="PATH|auto|none", help="record the run's harness tree (its "
                   "CLAUDE.md, AGENTS.md, skills, hooks, MCP servers and other parts) in run.ext: PATH, or auto for the "
                   "git working tree the attempts ran in; default: config harness.record, which is none")
    q.add_argument("--no-fit", action="store_true", help="do not start a background refit")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_finish"))

    q = rs.add_parser("import", help="store OCP v0.3 runs written by the orchestrator (files or a directory)")
    q.add_argument("files", nargs="+", metavar="FILE.ocp.json|DIR", help="OCP documents, or a directory of *.ocp.json files")
    q.add_argument("--finish", action="store_true", help="also finish each run once stored; one background refit at the end")
    q.add_argument("--no-fit", action="store_true", help="do not start the background refit")
    q.add_argument("--replace", action="store_true", help="take this copy of a run already in the store with different "
                   "content (without it such a run is refused); outcomes added after the import are kept, and a "
                   "copy that changes one is refused")
    q.add_argument("--brief", action="store_true", help="with --json: a short object for agents (counts, failures with "
                   "file and error, the overlap with the shipped prior, the next step)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_import"))

    q = rs.add_parser("record", help="record a finished run in one call: its sessions, commits and verdicts, then finish")
    q.add_argument("--run", default=None, metavar="RUN", help="the open run from run start; without it a run is opened "
                   "from the flags below, as run start does")
    _task_args(q)
    q.add_argument("--rec", default=None, metavar="REC", help="the recommend id the run follows (with --choice, or --config)")
    q.add_argument("--choice", default=None, metavar="KEY", help="open the run from this choice of --rec (not pair)")
    q.add_argument("--config", default=None, metavar="CFG", help="configuration id from recommend")
    q.add_argument("--workflow", default=None, metavar="FILE.toml", help="a workflow file, catalog name or stored workflow")
    q.add_argument("--set", action="append", default=[], metavar="PIECE=HARNESS:MODEL:EFFORT", help="piece setting (repeatable)")
    q.add_argument("--source", default=None, choices=SOURCES, help="why this configuration was chosen (with --config or --workflow)")
    q.add_argument("--rule", default=None, metavar="RULE", help="a named acceptance rule from config; with --rec "
                   "it replaces the rule the recommendation was made for")
    q.add_argument("--session", action="append", default=[], metavar="[PIECE=]ID", help="a session that worked on the run: a "
                   "Claude Code session id, a Codex thread id, or self for this Claude Code session (repeatable); "
                   "PIECE= names its piece; run sessions lists the ids. Only the part of a session inside the window "
                   "counts; a Codex session counts the usage it logged inside the window: each reading's increase at "
                   "the reading's time, a reading on either edge included. A sub-agent or Codex child thread counts in "
                   "full when it started inside the window, not at all when it started before; a Codex reading lower "
                   "than the one before it (a compaction reset) counts from zero; a Codex session with a reading that "
                   "has no time, or whose increases inside the window add up to less than zero, counts whole, with a note")
    q.add_argument("--cwd", action="append", default=[], metavar="[PIECE=]PATH", help="a folder an agent with no session id "
                   "worked in; its log is matched at finish by folder and time (repeatable)")
    q.add_argument("--since", default=None, metavar="TS", help="when the work began (e.g. 2h or a local ISO time); with "
                   "--run the default is the run's start")
    q.add_argument("--until", default=None, metavar="TS", help="when the work ended (e.g. 1h or a local ISO time; default now)")
    q.add_argument("--verified", action="append", default=[], metavar="NAME=VALUE", help="a verdict you observed yourself "
                   "(pass, fail, accept, reject, error)")
    q.add_argument("--reported", action="append", default=[], metavar="NAME=VALUE", help="a verdict someone reported")
    q.add_argument("--score", action="append", default=[], metavar="NAME=VALUE", help="a score (NAME= when not measured)")
    commits = q.add_mutually_exclusive_group()
    commits.add_argument("--commit", action="append", default=[], metavar="SHA", help="a commit the run made (repeatable)")
    commits.add_argument("--no-commits", action="store_true", help="do not record the commits since the base commit")
    q.add_argument("--harness", default=None, metavar="PATH|auto|none", help="record the run's harness tree (its "
                   "CLAUDE.md, AGENTS.md, skills, hooks, MCP servers and other parts) in run.ext: PATH, or auto for the "
                   "git working tree the attempts ran in; default: config harness.record, which is none")
    q.add_argument("--no-fit", action="store_true", help="do not start a background refit")
    _check_args(q)
    _common(q)
    q.set_defaults(slate=None, new_slate=False, func=_lazy("loopmath.store.record:run_record"))

    q = rs.add_parser("sessions", help="list the recent Claude Code and Codex sessions in this folder, with their ids")
    q.add_argument("--cwd", default=None, metavar="PATH", help="the folder the sessions ran in (default: this one)")
    q.add_argument("--days", type=float, default=7, metavar="N", help="how far back to look (default 7)")
    q.add_argument("--limit", type=int, default=10, metavar="N", help="at most this many, newest first (default 10)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:run_sessions"))

    p = sub.add_parser("outcome", help="record a verdict, score, late event or pair preference")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", default=None, metavar="RUN", help="the run the signal is about")
    target.add_argument("--slate", default=None, metavar="SLT", help="the slate a preference is about (with --prefer)")
    target.add_argument("--commit", default=None, metavar="SHA", help="late event: find runs by commit")
    target.add_argument("--rule", default=None, metavar="NAME",
                        help="review: judge each RUN by its latest review or accepted verdict (accept, approve or "
                        "pass is a success); --undo-rule undoes that, and so does naming the rule a run had before "
                        "(tests when it had none) unless it is named review. A run with a tests verdict is refused, "
                        "and nothing is written if any RUN is refused")
    target.add_argument("--undo-rule", action="store_true",
                        help="restore each RUN's rule from before --rule review moved it, exactly as it was")
    p.add_argument("rule_runs", nargs="*", default=[], metavar="RUN", help="with --rule or --undo-rule: the runs to move")
    p.add_argument("--signal", default=None, metavar="NAME=VALUE", help="NAME= (empty) declares an unmeasured score")
    p.add_argument("--kind", default="verdict", choices=["verdict", "score", "event"], help="what the signal is (default verdict)")
    p.add_argument("--unit", default=None, metavar="U", help="score unit")
    p.add_argument("--better", default=None, choices=["higher", "lower"], help="which way a score improves")
    p.add_argument("--target", type=float, default=None, metavar="X", help="the score that counts as success")
    p.add_argument("--scale", default="linear", choices=["linear", "log", "fraction"], help="how the score is modelled")
    p.add_argument("--at-attempt", default=None, metavar="ATT", help="the attempt the signal belongs to")
    p.add_argument("--source", default="orchestrator", choices=["orchestrator", "user", "ci", "tracker", "judge"],
                   help="who reports the signal")
    p.add_argument("--tier", default="reported", choices=TIER_CHOICES, help="how far the signal can be trusted")
    p.add_argument("--observed-at", default=None, metavar="TS", help="ISO time it was observed (default: now)")
    p.add_argument("--prefer", default=None, metavar="RUN|tie", help="the slate member that did better, or tie")
    p.add_argument("--judge", default=None, choices=["referee", "user", "model"], help="who judged the preference (default referee)")
    p.add_argument("--blinded", action="store_true", help="the judge did not know which run was which")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.store.commands:outcome"))

    p = sub.add_parser("budget", help="spend against the cap; set the cap")
    p.add_argument("--usd", type=float, default=None, metavar="X", help="set the cap in dollars")
    p.add_argument("--period", default=None, choices=["week", "month", "none"], help="the cap is per week, per month, or in total (none)")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.store.commands:budget"))

    p = sub.add_parser("config", help="get or set configuration")
    cs = p.add_subparsers(dest="config_command", required=True)
    q = cs.add_parser("get", help="print one key, or all of the config")
    q.add_argument("key", nargs="?", default=None, metavar="KEY")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:config_get"))
    q = cs.add_parser("set", help="set one key",
                      description="Sets one key in config.toml. benchmark_prior_weight is unset by default: each "
                                  "benchmark in the shipped benchmarks.toml has its own weight, in runs, shared by "
                                  "a model's results on it; a number here is one weight for every benchmark, and 0 "
                                  "turns benchmark priors off.")
    q.add_argument("key", metavar="KEY")
    q.add_argument("value", metavar="VALUE")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:config_set"))
    q = cs.add_parser("unset", help="remove one key, so it reads its default",
                      description="Removes one key from config.toml and keeps every other line, comments too. "
                                  "A key that is not in the file is not an error.")
    q.add_argument("key", metavar="KEY")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.store.commands:config_unset"))


LABELER_KINDS = ("claude", "codex", "command")


def labeler_spec(value: str) -> str:
    """`--labeler`: `claude:<model>`, `codex:<model>`, `command:<cmd>` or `none`."""
    kind, sep, rest = value.partition(":")
    if value == "none" or (sep and kind in LABELER_KINDS and rest.strip()):
        return value
    raise argparse.ArgumentTypeError(
        f"expected claude:<model>, codex:<model>, command:<cmd> or none, not {value!r}"
    )


# ---------------------------------------------------------------- learning
def _add_learning(sub) -> None:
    p = sub.add_parser("fit", help="refit the belief from the prior bundle and your runs",
                       description="Builds the belief from the shipped prior and your runs and points fits/latest at it: "
                                   "a new fit replaces the one recommend and posterior read. The 5 newest fits, and any a "
                                   "receipt or stored recommendation names, are kept; recommend --fit ID and posterior "
                                   "--fit ID read one of them. Runs in your store always count as yours.")
    p.add_argument("--background", action="store_true", help="start the fit and return at once")
    p.add_argument("--full", action="store_true", help="also run the PyMC check (needs the bayes extra)")
    p.add_argument("--no-prior", action="store_true", help="fit your runs alone, without the shipped prior, shared runs or benchmark factors")
    p.add_argument("--without", action="append", default=[], metavar="SOURCE",
                   help="leave out one source: e0, sweep or rq1 (shipped prior; never your own runs, whatever their label), benchmark (benchmark prior factors), user (your runs), shared or shared:ORG (imported runs); repeatable")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.belief.commands:fit"))

    p = sub.add_parser("onboard", help="turn your Claude Code and Codex history into runs and a first fit",
                       description="Reads your Claude Code and Codex history (read only), groups sessions, has the labeller you choose "
                                   "name each task type, records one habit run per group, saves your usual workflow per task type "
                                   "and repo, and runs a first fit. Reading the history uses up to 8 worker processes "
                                   "(one per CPU); set LOOPMATH_WORKERS=1 to keep it in one process.")
    p.add_argument("--since", default="90d", metavar="D", help="how far back to read: 90d, 12w, 36h or a date (default 90d)")
    p.add_argument("--labeler", default=None, type=labeler_spec, metavar="claude:MODEL|codex:MODEL|command:CMD|none",
                   help="what labels task types: a model through your own claude or codex CLI, your own command, or none; there is no default, and your choice is saved as config onboard.labeler")
    p.add_argument("--yes", action="store_true", help="go ahead without asking, after showing the expected labelling cost")
    p.add_argument("--dry-run", action="store_true", help="show what it would do: no labelling calls, no runs or fit written")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.onboard.commands:onboard"))

    p = sub.add_parser("share", help="write the shareable part of your store to a file",
                       description="Keeps task types, features, workflows, settings, tokens, dollars and outcomes; "
                                   "repos, tasks and runs only as salted hashes; no titles, paths, commands, session ids or commits.")
    p.add_argument("--out", default=None, metavar="FILE", help="the share file to write: plain JSON for FILE.json, gzip JSON for FILE.json.gz or any other name (check it first with --preview)")
    p.add_argument("--since", default=None, metavar="D", help="only runs since then: 90d, 4w, 12h or a date")
    p.add_argument("--preview", action="store_true", help="print exactly what would leave; writes no share file")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.share.commands:share"))

    p = sub.add_parser("prior", help="the shipped prior bundle: show, import-shared, remove-shared")
    ps = p.add_subparsers(dest="prior_command", required=True, metavar="{show,import-shared,remove-shared}")
    # Hidden, as `plan` is: no help= keeps `build` out of `loopmath prior --help`; it stays callable by name.
    q = ps.add_parser("build", description="Rebuild the packaged prior bundle from the sweep, E0 and RQ1 inputs. "
                                           "It needs research data folders that do not ship with the package.")
    q.add_argument("--out", default=None, metavar="DIR", help="where to write the bundle (required)")
    q.add_argument("--sweep-dir", action="append", default=None, metavar="PATH", help="sweep results, one folder per batch; repeatable (default: LOOPMATH_SWEEP_DIR, one folder)")
    q.add_argument("--e0-corpus", default=None, metavar="PATH", help="E0 corpus (default: LOOPMATH_E0_CORPUS)")
    q.add_argument("--rq1-dir", default=None, metavar="PATH", help="RQ1 OCP documents (default: LOOPMATH_PRIOR_RQ1)")
    q.add_argument("--lanes-dir", default=None, metavar="PATH", help="build lane rows (default: LOOPMATH_PRIOR_LANES)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.priors.commands:build"))
    q = ps.add_parser("show", help="what the packaged prior bundle holds and where it came from")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.priors.commands:show"))
    q = ps.add_parser("import-shared", help="add a share file from another organization to your priors")
    q.add_argument("file", metavar="FILE", help="a loopmath.share/1 file written by loopmath share (gzip or plain JSON)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.share.commands:import_shared"))
    q = ps.add_parser("remove-shared", help="take an imported organization's runs out of your priors")
    q.add_argument("org", metavar="ORG", help="the organization import-shared named, as shared:HASH or HASH")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.share.commands:remove_shared"))


# ---------------------------------------------------------------- viewing
def _add_viewing(sub) -> None:
    p = sub.add_parser("runs", help="previous runs (table, one run, HTML page)")
    p.add_argument("run_id", nargs="?", default=None, metavar="RUN", help="show this one run in full, as --run RUN")
    p.add_argument("--type", dest="task_type", default=None, metavar="T", help="only this task type (see loopmath task-types)")
    p.add_argument("--repo", default=None, metavar="R", help="only this repository")
    p.add_argument("--since", default=None, metavar="D", help="only runs started since then: 90d, 2w, 12h, or a date or time such as 2026-01-15 or 2026-01-15T09:30 (local time)")
    p.add_argument("--slate", default=None, metavar="SLT", help="only this slate")
    p.add_argument("--run", default=None, metavar="RUN", help="show this one run in full; RUN is an id from the run column of loopmath runs")
    p.add_argument("--wide", action="store_true", help="also show each run's cfg_ configuration id")
    _common(p, html=True)
    p.set_defaults(func=_lazy("loopmath.views.runs:command"))

    p = sub.add_parser("posterior", help="current estimates by level and by workflow graph")
    p.add_argument("--level", default="all",
                   choices=["model", "effort", "role", "topology", "type", "repo", "feature", "harness", "all"], help="which level of estimates (default all; harness: the harness labels and their parts, experimental)")
    p.add_argument("--head", default=None, metavar="cost|tokens|success|gate|score:NAME", help="which quantity to show")
    p.add_argument("--target", default=None, metavar="NAME>=X", help="score target for the chance to reach and the cost per accepted result, as for recommend (default: the latest recommendation's target for this type and repo)")
    p.add_argument("--workflow", default=None, metavar="CFG|FILE.toml", help="also draw this configuration with each piece's estimate: a cfg_ id from loopmath recommend --json or your runs, or FILE.toml with settings")
    p.add_argument("--type", dest="task_type", default=None, metavar="T", help="task type for the workflow graph (default: the most common in your runs, else feature)")
    p.add_argument("--repo", default=None, metavar="R", help="repository for the workflow graph (default: the most common in your runs)")
    p.add_argument("--subtype", default=None, metavar="S", help="your own subtype of the task type, as for recommend")
    p.add_argument("--feature", action="append", default=[], metavar="K=V", help="task feature for the workflow graph (repeatable), as for recommend")
    p.add_argument("--horizon", default=None, metavar="TIME", help=HORIZON_HELP)
    p.add_argument("--fit", default=None, metavar="ID", help=FIT_HELP)
    p.add_argument("--all", action="store_true", help="also show the rows about providers and models you have not run, and every row")
    _common(p, html=True)
    p.set_defaults(func=_lazy("loopmath.views.posterior:command"))

    p = sub.add_parser("status", help="fit age, open runs, spend, store health")
    p.add_argument("--reindex", action="store_true", help="rebuild the run index from the run files first")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.store.commands:status"))

    p = sub.add_parser("report", help="receipt calibration and exploration outcomes")
    p.add_argument("--since", default=None, metavar="D", help="only receipts since then: 30d, 2w, 12h or a date")
    _common(p, html=True)
    p.set_defaults(func=_lazy("loopmath.store.commands:report"))

    p = sub.add_parser("doctor", help="check logs, store, fit, prices and skill")
    _common(p)
    p.set_defaults(func=_lazy("loopmath.skill.commands:doctor"))

    p = sub.add_parser("skill", help="install, remove or print the orchestrator skills")
    ss = p.add_subparsers(dest="skill_command", required=True)
    for verb, helptext in (("install", "write the six skills for Claude Code, Codex or both"),
                           ("uninstall", "remove what install wrote")):
        q = ss.add_parser(verb, help=helptext)
        q.add_argument("--target", default="auto", choices=["auto", "claude-code", "codex", "both"],
                       help="which agent: auto (default) is every agent whose home folder exists, else claude-code")
        q.add_argument("--scope", default="user", choices=["user", "project"],
                       help="for you in every folder (user), or for this repository only (project)")
        q.add_argument("--dir", default=None, metavar="PATH", help="the project folder for --scope project "
                       "(default: this folder)")
        if verb == "install":
            q.add_argument("--dry-run", action="store_true", help="print every file install would write or "
                           "change, and write nothing")
        _common(q)
        q.set_defaults(func=_lazy(f"loopmath.skill.commands:{verb}"))
    q = ss.add_parser("show", help="print one skill (default loopmath, the start-here skill) or the shared reference")
    q.add_argument("name", nargs="?", default="loopmath", metavar="NAME",
                   help="loopmath, loopmath-onboard, loopmath-import-runs, loopmath-update-fit, loopmath-plan-task, "
                   "loopmath-record-run, or reference")
    q.set_defaults(func=_lazy("loopmath.skill.commands:show"))

    p = sub.add_parser("ocp", help="validate or migrate OCP documents")
    os_ = p.add_subparsers(dest="ocp_command", required=True)
    q = os_.add_parser("validate", help="check OCP documents against the schema and conformance rules (exit 1 on an error, 2 on a missing file)")
    q.add_argument("files", nargs="+", metavar="FILE")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.ocp.commands:validate"))
    q = os_.add_parser("migrate", help="convert OCP v0.1 and v0.2 documents and contract v3 run files to v0.3 (inputs are never changed)")
    q.add_argument("files", nargs="+", metavar="FILE")
    q.add_argument("--out", default=None, metavar="DIR", help="write each result to DIR as NAME.ocp.json; a result with errors is never written (needed for several files unless --json)")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.ocp.commands:migrate"))


HIDDEN = frozenset({"plan", "analyze-e0", "scan", "research"})  # callable, but not listed by `loopmath --help`


def command_metavar(sub) -> str:
    """The `{a,b,...}` command list for usage lines, without the hidden commands."""
    return "{" + ",".join(name for name in sub.choices if name not in HIDDEN) + "}"


def _add_gate(sub) -> None:
    p = sub.add_parser("gate", help="experimental: adopt or reject a harness change: compare base and candidate runs",
                       description="Experimental: adopt or reject a harness change, comparing base and candidate "
                                   "runs, for a loop that adopts changes on its own.")
    p.add_argument("--base", default=None, metavar="SEL",
                   help="the base arm, required (as --cand is) unless a subcommand runs: digest:sha256:<hex> or "
                        "sha256:<hex> (the harness record's digest; a prefix of "
                        "one digest will do), a cfg_ configuration id, or arm:<name> or arm=<name>")
    p.add_argument("--cand", default=None, metavar="SEL", help="the candidate arm, in the same forms as --base")
    p.add_argument("--runs", default=None, metavar="FILE",
                   help="read loopmath.harness-runs/0.1 rows from FILE instead of the store's runs; the store's "
                        "config is still read for its rules")
    p.add_argument("--task", default=None, metavar="ID", help="only runs of this task")
    p.add_argument("--rule", default="score:judge", metavar="R",
                   help="what counts as better: score:NAME (higher is better), score:NAME:lower, an acceptance rule "
                        "such as accept, or a rule defined under [rules.NAME] in the store's config (default "
                        "score:judge)")
    p.add_argument("--floor", type=float, default=None, metavar="X",
                   help="the smallest gain worth adopting: in the score's units under a score rule, a fraction under "
                        "an acceptance rule (default: compare's, 0 for a score)")
    p.add_argument("--r-min", type=int, default=2, metavar="N", help="pairs before any adoption (default 2)")
    p.add_argument("--r-max", type=int, default=8, metavar="N", help="most replicates per arm (default 8)")
    p.add_argument("--budget-usd", type=float, default=None, metavar="X",
                   help="cap on the dollars of both arms' runs, 0 or more (0 buys no further pair); needs every "
                        "run's cost")
    p.add_argument("--max-cost-ratio", type=float, default=None, metavar="R",
                   help="reject when the candidate costs more than R times the base per run; needs every run's cost")
    p.add_argument("--value-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="the value stop, on by default: reject, keeping the base, from --min-pairs scored pairs on, "
                        "only when the next pairs are certified not worth their price: every plan's net value below "
                        "0 by more than two standard errors, or no look left can change the action; an unresolved "
                        "value never stops, nor does a cost that is not finite. At price 0 it fires only when going "
                        "on is certified to lose, or when no look left can change the action; a sample with no "
                        "adoption never fires. --no-value-stop turns it off: the next pairs' value and price are "
                        "then shown and stop nothing, as in check-change without --value-stop")
    p.add_argument("--min-pairs", type=int, default=None, metavar="K",
                   help="the scored pairs before the value may stop the gate (default 4; 1 stops from the first "
                        "pair); refused with --no-value-stop")
    p.add_argument("--pair-price", type=float, default=None, metavar="USD",
                   help="the price of one pair that the next pairs' value is compared with (default: its expected "
                        "cost); 0 for runs on a subscription: the budget still counts the runs' recorded dollars")
    p.add_argument("--horizon-runs", type=float, default=None, metavar="N",
                   help="the number of future runs the change will serve, which prices the next pairs (default 50)")
    p.add_argument("--value-usd", type=float, default=None, metavar="USD", help="dollars per point of gain "
                   "(default: the base's cost per accepted result)")
    _common(p, home_help=HOME_HELP + "; the store's config.toml is read for the rules it defines, also with --runs")
    # --base and --cand are required unless a subcommand runs; the handler says so as argparse would
    p.set_defaults(func=_lazy("loopmath.gate.commands:gate"), usage_error=p.error)
    _add_gate_winner(p)


def _add_gate_winner(gate: argparse.ArgumentParser) -> None:
    sub = gate.add_subparsers(dest="gate_command", metavar="winner", title="the search's winner",
                              description="`gate winner` asks how much of a search loop's best score is real: run "
                                          "`loopmath gate winner --help`.")
    p = sub.add_parser("winner", help="experimental: how much of a search loop's winning lead is real, and what to "
                                      "run before adopting it",
                       description="A loop that scores many candidates and keeps the best overstates the best "
                                   "one's gain: its noise was lucky. gate winner shrinks the winner's lead with a "
                                   "prior learnt from all the candidates' scores, reports the gain with its 80% and "
                                   "95% ranges and P(beats parent), and runs the gate on the confirmatory runs "
                                   "(--confirm) to say what to run next. It reads no store.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", default=None, metavar="FILE",
                     help="the search's scores: a CSV with a header (id, score; optional se, reps, order, role "
                          "parent|candidate, at_most, at_least, count for unscored candidates) or a "
                          "loopmath.gate-winner-scores/0.1 JSON file")
    src.add_argument("--runs", default=None, metavar="FILE",
                     help="the search's runs as loopmath.harness-runs/0.1 rows: each arm (harness digest, else arm, "
                          "else configuration) is a candidate, scored by its mean over the tasks every arm ran")
    p.add_argument("--parent", default="arm:base", metavar="SEL",
                   help="with --runs, the arm the loop started from, in gate's selector forms (default arm:base)")
    p.add_argument("--winner", default=None, metavar="ID",
                   help="the candidate the loop picked, when it is not the best scored one")
    p.add_argument("--rule", default="score:judge", metavar="R",
                   help="the score: score:NAME (higher is better) or score:NAME:lower (default score:judge)")
    p.add_argument("--spread", default=None, metavar="X|LO:HI",
                   help="one evaluation's SD across reruns on the same tasks, in the score's units, or a range "
                        "LO:HI to integrate over when it is not measured; with --runs, one run's SD on one task. "
                        "Needed when the scores have no se and the runs no replicate")
    p.add_argument("--confirm", default=None, metavar="FILE",
                   help="confirmatory runs of the parent and the winner, as loopmath.harness-runs/0.1 rows, for "
                        "the gate; with --scores each row is one complete evaluation")
    p.add_argument("--confirm-base", default="arm:base", metavar="SEL",
                   help="the parent's confirmatory runs in --confirm (default arm:base)")
    p.add_argument("--confirm-cand", default="arm:candidate", metavar="SEL",
                   help="the winner's confirmatory runs in --confirm (default arm:candidate)")
    p.add_argument("--floor", type=float, default=None, metavar="X",
                   help="the smallest gain worth adopting, in the score's units (default 0)")
    p.add_argument("--r-min", type=int, default=2, metavar="N", help="pairs before any adoption (default 2)")
    p.add_argument("--r-max", type=int, default=8, metavar="N", help="most confirmatory runs per arm (default 8)")
    p.add_argument("--budget-usd", type=float, default=None, metavar="X",
                   help="cap on the dollars of both arms' confirmatory runs; needs every run's cost")
    p.add_argument("--prior", default="sparse", choices=("sparse", "ridge", "auto"),
                   help="the candidates' spread around the parent: most near zero and a share spread out (sparse, "
                        "the default), normal (ridge), or both weighed by the scores (auto)")
    p.add_argument("--draws", type=int, default=200, metavar="N",
                   help="draws for the projection of what the gate will do (default 200; 0: none)")
    p.add_argument("--seed", type=int, default=0, metavar="N", help="the projection's seed (default 0)")
    _common(p, home_help="accepted for the common form; gate winner reads no store")
    p.set_defaults(func=_lazy("loopmath.gate.winner_cli:gate_winner"))


def _add_tasks(sub) -> None:
    p = sub.add_parser("tasks", help="experimental: score a candidate on fewer tasks: the next task, an estimate, "
                                     "stop or continue",
                       description="Experimental: score a candidate harness on a subset of a task suite, learning "
                                   "each task from other arms' runs.")
    ts = p.add_subparsers(dest="tasks_command", required=True, metavar="{pick}")
    q = ts.add_parser("pick", help="the next tasks for a candidate, its suite score so far, and stop or continue",
                      description="Experimental: from the history (other arms' runs on the suite's tasks) and the "
                                  "candidate's runs so far, the next tasks to run, the candidate's suite score with "
                                  "its 80% and 95% ranges, and stop or continue. Each call refits; from a store it "
                                  "writes each next task's file under <home>/tasks-pick/ and prints the run record "
                                  "line that records the run on it, in the same store. The ranges assume the "
                                  "candidate behaves like the history's arms: for one unlike all of them (a new "
                                  "model rather than a harness change on a known one) they can be much too narrow "
                                  "under loss, info or fixed, and --method random or stratified keeps them closer "
                                  "to honest; a candidate scoring below every history arm is estimated too high by "
                                  "every method.")
    q.add_argument("--candidate", required=True, metavar="SEL",
                   help="the candidate's runs: KEY=VALUE for a run label (the --label of run record), arm=<name>, "
                        "a cfg_ configuration id, or digest:sha256:<hex>")
    q.add_argument("--runs", action="append", default=None, metavar="FILE",
                   help="read loopmath.harness-runs/0.1 rows from FILE (repeatable) instead of the store's runs; "
                        "no task file is written")
    q.add_argument("--history", action="append", default=None, metavar="SEL",
                   help="only these arms' runs as the history (repeatable; default every run not the candidate's)")
    q.add_argument("--suite", default=None, metavar="FILE",
                   help="the suite's task ids, one per line (a task JSON object, or the id); default every task "
                        "the history or the candidate has run")
    q.add_argument("--rule", default=None, metavar="R",
                   help="what counts as a pass: accept (each run's own rule, the default), a rule name or short "
                        "form, or score:NAME for a score from 0 to 1")
    q.add_argument("--n", type=int, default=1, metavar="N", help="how many next tasks (default 1)")
    q.add_argument("--method", default="loss", choices=("loss", "info", "fixed", "random", "stratified"),
                   help="how the next task is chosen: loss (the largest expected cut in the score's variance, the "
                        "default), info (Fisher information at the current ability), fixed (one order for every "
                        "candidate), random, stratified; random or stratified for a candidate unlike the "
                        "history's arms")
    q.add_argument("--model", default="auto", choices=("auto", "rasch", "2pl"),
                   help="the task model: rasch, 2pl (a slope per task) or auto (2pl from 8 history arms on)")
    stop = q.add_mutually_exclusive_group()
    stop.add_argument("--halfwidth", type=float, default=None, metavar="H",
                      help="stop once the 80%% range is within +-H (default: what a full run of the suite gives)")
    stop.add_argument("--versus", type=float, default=None, metavar="X",
                      help="stop once the score is above or below X at --level")
    stop.add_argument("--versus-arm", default=None, metavar="SEL",
                      help="stop once the score is above or below this arm's, at --level")
    q.add_argument("--level", type=float, default=0.9, metavar="P", help="the versus stop's level (default 0.9)")
    q.add_argument("--min-sessions", type=int, default=5, metavar="K", help="no stop before K sessions (default 5)")
    q.add_argument("--max-sessions", type=int, default=None, metavar="M", help="stop at M sessions")
    q.add_argument("--seed", type=int, default=0, metavar="S", help="the random and stratified orders' seed")
    _common(q)
    q.set_defaults(func=_lazy("loopmath.tasks.commands:tasks_pick"))


HARNESS_DESCRIPTION = """Experimental: tune a Claude Code harness on your own task. Editor sessions propose edits to the
repo's CLAUDE.md, skills and agents; a screening design of task runs measures each edit against your score;
the kept edits are delivered only when they beat the base. Every config field has a flag."""

HARNESS_FLOW = """flow:
  loopmath harness init                    write loopmath-harness.toml
  loopmath harness plan                    sessions, cost point and range, time, preflight; no model call
  loopmath harness run --detach --yes      start a tune once its spend is agreed
  loopmath harness watch ID                follow it to the end
  loopmath harness show ID                 what was delivered, and why
  loopmath harness apply ID                write the delivered harness into the repo; commits nothing

exit codes: 0 delivered, 1 error, 2 usage, 3 refused by budget or preflight, 4 not enough signal (the base is
delivered), 5 stopped, 6 paused (resume continues)"""

HARNESS_VERBS = ("init", "plan", "run", "status", "watch", "stop", "resume", "show", "apply", "list", "config",
                 "schema", "skill", "mod", "herdr")


HARNESS_FLAGS = "loopmath.harness.tune.commands:add_config_flags"


def _late_flags(base: type) -> type:
    """A parser class whose `flags=` target (a `_lazy` string) adds its flags the first time the parser parses or
    prints help, so the config flags, generated from the tune's field table, load only for those verbs."""

    class LateFlags(base):
        def __init__(self, *a, flags: str | None = None, **kw):
            super().__init__(*a, **kw)
            self._late = flags

        def _load(self) -> None:
            target, self._late = self._late, None
            if target:
                _lazy(target)(self)

        def parse_known_args(self, args=None, namespace=None):
            self._load()
            return super().parse_known_args(args, namespace)

        def format_help(self):
            self._load()
            return super().format_help()

        def format_usage(self):
            self._load()
            return super().format_usage()

    return LateFlags


def _add_harness(sub) -> None:
    target = "loopmath.harness.tune.commands:"
    p = sub.add_parser("harness", help="experimental: tune a Claude Code harness: init, plan, run, watch, show, "
                       "apply", description=HARNESS_DESCRIPTION, epilog=HARNESS_FLOW,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    hs = p.add_subparsers(dest="harness_command", required=True, metavar="{" + ",".join(HARNESS_VERBS) + "}",
                          parser_class=_late_flags(type(p)))

    def ident(q, nargs="?", text="the tune: its id, the end of its id, or its folder (default: the newest)"):
        q.add_argument("id", nargs=nargs, default=None, metavar="ID", help=text)

    q = hs.add_parser("init", flags=HARNESS_FLAGS,
                      help="write loopmath-harness.toml with the defaults, one comment per field",
                      description="Write the config file: the defaults and any flags, one comment per field.")
    q.add_argument("--out", default=None, metavar="PATH", help="where to write it (default ./loopmath-harness.toml)")
    q.add_argument("--force", action="store_true", help="overwrite an existing file")
    _common(q)
    q.set_defaults(func=_lazy(target + "init_verb"))

    q = hs.add_parser("plan", flags=HARNESS_FLAGS,
                      help="the dry run: stages, sessions, cost point and range, time, preflight",
                      description="The dry run, with no model call: the resolved stages and their order, session "
                                  "counts by component, the cost point and range, the worst-case time, the "
                                  "design's residual df, the signal and budget warnings, and the preflight checks.")
    _common(q)
    q.set_defaults(func=_lazy(target + "plan_verb"))

    q = hs.add_parser("run", flags=HARNESS_FLAGS,
                      help="start a tune and print its id (spends model calls)",
                      description="Start a tune. It prints the plan and asks on a terminal; without one it "
                                  "refuses unless --yes. With --detach the engine runs as its own session and the "
                                  "command returns at once.")
    q.add_argument("--detach", action="store_true", help="return at once; the engine runs on in its own session")
    q.add_argument("--yes", action="store_true", help="the spend is agreed: do not ask")
    q.add_argument("--parts", default=None, help=argparse.SUPPRESS)
    _common(q)
    q.set_defaults(func=_lazy(target + "run_verb"))

    q = hs.add_parser("status", help="a snapshot: stage, sessions, spend against the caps, last events",
                      description="A snapshot of a tune, folded from its event stream.")
    ident(q)
    q.add_argument("--line", action="store_true", help="one line, as the Claude Code mod's band shows it")
    _common(q)
    q.set_defaults(func=_lazy(target + "status_verb"))

    q = hs.add_parser("watch", help="follow a tune's events until it ends",
                      description="Follow a tune's events.jsonl until tune.end (or a pause with no engine). The "
                                  "exit code is the tune's.")
    ident(q)
    q.add_argument("--from", dest="from_seq", type=int, default=1, metavar="SEQ", help="start at this seq")
    q.add_argument("--pretty", action="store_true", help="readable lines (the default without --json)")
    q.add_argument("--poll", type=float, default=0.5, help=argparse.SUPPRESS)
    q.add_argument("--grace", type=float, default=10.0, help=argparse.SUPPRESS)
    _common(q, json_flag=False)
    q.add_argument("--json", action="store_true", help="the raw events, one JSON object per line")
    q.set_defaults(func=_lazy(target + "watch_verb"))

    q = hs.add_parser("stop", help="ask a tune to stop: no new session starts",
                      description="Write the stop file. The engine finishes the running sessions, or with --now "
                                  "kills them by recorded pid, then ends stopped.")
    ident(q)
    q.add_argument("--now", action="store_true", help="kill the running sessions by recorded pid")
    _common(q)
    q.set_defaults(func=_lazy(target + "stop_verb"))

    q = hs.add_parser("resume", help="continue a paused or interrupted tune (spends model calls)",
                      description="Continue a tune from its state.json. A session with no end record runs again "
                                  "from a fresh worktree.")
    ident(q, nargs=None, text="the tune: its id, the end of its id, or its folder")
    q.add_argument("--detach", action="store_true", help="return at once; the engine runs on in its own session")
    q.add_argument("--parts", default=None, help=argparse.SUPPRESS)
    _common(q)
    q.set_defaults(func=_lazy(target + "resume_verb"))

    q = hs.add_parser("show", help="the delivered harness and why; the effects table; spend by component",
                      description="What a tune delivered and why, with the effects table.")
    ident(q)
    q.add_argument("--diff", action="store_true", help="the delivered harness against the base")
    q.add_argument("--effects", action="store_true", help="the effects table")
    q.add_argument("--ledger", action="store_true", help="spend by role and model, beside Claude Code's own figure")
    _common(q)
    q.set_defaults(func=_lazy(target + "show_verb"))

    q = hs.add_parser("apply", help="write the delivered harness into the repo; commits nothing",
                      description="Write the delivered harness files into the repo's working tree (the hooks key "
                                  "only of .claude/settings.json). Refuses when those paths have uncommitted "
                                  "changes. Commits nothing.")
    ident(q)
    q.add_argument("--to", default=None, metavar="REPO", help="this working tree (default: the tune's repo)")
    q.add_argument("--work", action="store_true", help="also apply the delivered run's diff")
    q.add_argument("--dry-run", action="store_true", help="say what would be written, and write nothing")
    _common(q)
    q.set_defaults(func=_lazy(target + "apply_verb"))

    q = hs.add_parser("list", help="tunes, newest first", description="The tunes in the store, newest first.")
    _common(q)
    q.set_defaults(func=_lazy(target + "list_verb"))

    q = hs.add_parser("config", help="read or edit loopmath-harness.toml in place, keeping comments",
                      description="Read or edit the config file in place. Comments, unknown keys and the order "
                                  "stay; every value is checked against its field.")
    cs = q.add_subparsers(dest="harness_config_command", required=True, metavar="{get,set,unset}")
    c = cs.add_parser("get", help="a value, or the whole config with --all",
                      description="A value (file, else default), or the whole effective config.")
    c.add_argument("key", nargs="?", default=None, metavar="KEY", help="a dotted key, such as budget.usd")
    c.add_argument("--all", action="store_true", help="the whole effective config, and its errors")
    c.add_argument("--config", default=None, metavar="PATH", help="the config file")
    _common(c)
    c.set_defaults(func=_lazy(target + "config_get_verb"))
    c = cs.add_parser("set", help="set one key", description="Set one key in the file, keeping comments.")
    c.add_argument("key", metavar="KEY", help="a dotted key: TABLE.KEY or editors.N.KEY")
    c.add_argument("value", metavar="VALUE", help="the value, read as TOML when it parses, else as text")
    c.add_argument("--config", default=None, metavar="PATH", help="the config file")
    _common(c)
    c.set_defaults(func=_lazy(target + "config_set_verb"))
    c = cs.add_parser("unset", help="remove one key (its default applies) or one editor block (editors.N)",
                      description="Remove one key from the file, or one editor block with editors.N.")
    c.add_argument("key", metavar="KEY", help="a dotted key, or editors.N")
    c.add_argument("--config", default=None, metavar="PATH", help="the config file")
    _common(c)
    c.set_defaults(func=_lazy(target + "config_unset_verb"))

    q = hs.add_parser("schema", help="the config's JSON Schema", description="The config's fields; with --json "
                      "its JSON Schema, which the Claude Code mod draws its configure pane from.")
    _common(q)
    q.set_defaults(func=_lazy(target + "schema_verb"))

    q = hs.add_parser("skill", help="the agent skill doc for driving this CLI",
                      description="The skill doc for an agent that drives `loopmath harness`.")
    q.add_argument("--format", default="md", choices=["md", "json"], help="markdown (default) or the same as data")
    _common(q)
    q.set_defaults(func=_lazy(target + "skill_verb"))

    q = hs.add_parser("mod", help="the Claude Code mod: install, uninstall, path, validate",
                      description="The Claude Code mod (/harness). install and uninstall print each claude "
                                  "command and ask before running it.")
    ms = q.add_subparsers(dest="harness_mod_command", required=True, metavar="{install,uninstall,path,validate}")
    for name, text in (("install", "copy the mod out of the package and add it to Claude Code"),
                       ("uninstall", "remove the mod from Claude Code")):
        m = ms.add_parser(name, help=text, description=text[0].upper() + text[1:] + ".")
        m.add_argument("--scope", default="user", choices=["user", "project"], help="Claude Code's scope "
                       "(default user)")
        m.add_argument("--print", dest="print_only", action="store_true", help="print the commands and run nothing")
        m.add_argument("--yes", action="store_true", help="run them without asking")
        _common(m)
        m.set_defaults(func=_lazy(target + "mod_verb"))
    m = ms.add_parser("path", help="the mod folder", description="The packaged mod folder, and the installed copy.")
    _common(m)
    m.set_defaults(func=_lazy(target + "mod_verb"))
    m = ms.add_parser("validate", help="claude plugin validate --strict on the mod, in a throwaway config folder",
                      description="Run `claude plugin validate --strict` on the packaged mod under a throwaway "
                                  "CLAUDE_CONFIG_DIR.")
    m.add_argument("--test", action="store_true", help="also run `claude plugin test` on it")
    _common(m)
    m.set_defaults(func=_lazy(target + "mod_verb"))

    q = hs.add_parser("herdr", help="the herdr plugin: link, unlink, path",
                      description="The herdr plugin (watch, status and stop actions). link and unlink print the "
                                  "herdr command and ask before running it.")
    hs2 = q.add_subparsers(dest="harness_herdr_command", required=True, metavar="{link,unlink,path}")
    for name, text in (("link", "link the packaged plugin into herdr"), ("unlink", "unlink it")):
        m = hs2.add_parser(name, help=text, description=text[0].upper() + text[1:] + ".")
        m.add_argument("--print", dest="print_only", action="store_true", help="print the command and run nothing")
        m.add_argument("--yes", action="store_true", help="run it without asking")
        _common(m)
        m.set_defaults(func=_lazy(target + "herdr_verb"))
    m = hs2.add_parser("path", help="the plugin folder", description="The packaged herdr plugin folder.")
    _common(m)
    m.set_defaults(func=_lazy(target + "herdr_verb"))

    # internal verbs: no help, so they stay out of the listing
    q = hs.add_parser("_drive")
    q.add_argument("spec")
    q.set_defaults(func=_lazy(target + "drive_verb"), json=False)
    q = hs.add_parser("_session")
    q.add_argument("tune")
    q.add_argument("label")
    q.add_argument("--pane", default="")
    q.add_argument("--parts", default=None)
    q.set_defaults(func=_lazy(target + "session_verb"), json=False)


def register(sub) -> None:
    """Add every 0.1.0 command to the top-level subparsers."""
    _add_planning(sub)
    _add_recording(sub)
    _add_learning(sub)
    _add_viewing(sub)
    _add_compare(sub)
    _add_screen(sub)
    importlib.import_module("loopmath.builder.cli").add(sub)  # `builder` (0.2.1)
    _add_gate(sub)
    _add_tasks(sub)
    _add_harness(sub)


# Handler targets, for the registry test: every one must import and be callable.
def targets(parser: argparse.ArgumentParser) -> list[str]:
    found: list[str] = []

    def walk(p: argparse.ArgumentParser) -> None:
        func = p.get_default("func")
        if func is not None and hasattr(func, "target"):
            found.append(func.target)
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    walk(child)

    walk(parser)
    return sorted(set(found))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0)
