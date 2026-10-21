"""`claude plugin eval` results read as `loopmath.harness-runs/0.1` rows (docs/plugin-eval.md, docs/adapters.md).

A run importer, not an ADE adapter: it registers nothing, and `loopmath adapt plugin-eval` routes to it before
the adapter lookup, as `adapt harness2` does. It reads files only. Claude Code is never run; every field below
was written from Anthropic's docs (https://code.claude.com/docs/en/plugin-evals; tests/plugin_eval/README.md
cites a URL for each).

The docs give no JSON example of the result, and they name neither the run's score member nor its grader
list. So the layout read here is provisional: a run's grade is its `score` (a number in [0, 1]) or the weighted
fraction of its one grader list, and every reading is checked against each case's documented
`aggregates.score` and `aggregates.delta`. A shape that admits two readings, or a reading that does not
reproduce the file's own numbers, is refused by name; the reader never fills in a grade the file does not give.

One row per run, replicates kept. Only a case with both arms and a delta is compared: the docs omit the delta
"when the case ran one arm or the arms aren't comparable", so any other case of a two-arm suite is withheld
from the rows, its grades kept in the summary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "claude-plugin-eval/1"
TOOL = "claude-plugin-eval"
NS = "dev.loopmath.adapter.plugin-eval"
VENDOR_NS = "com.anthropic.claude-code.plugin-eval"
SCHEMA_VERSIONS = (1,)
RESULT_FILE = "aggregate-result.json"
HARNESS_SCHEMA = "loopmath.harness/0.1"
ROWS_SCHEMA = "loopmath.harness-runs/0.1"
ARMS = ("with", "without")
DEFAULT_THRESHOLD = 1.0   # `--threshold` default (https://code.claude.com/docs/en/plugin-evals#command-options)
SCORE_NAME = "plugin-eval"
TOL = 1e-6
DOCS = "docs/plugin-eval.md"

# Every field of the result the reader uses; tests/plugin_eval/README.md cites a source for each.
FIELDS_USED = ("aggregate-result.json", "schemaVersion", "claudeVersion", "partial", "partialReason", "costUsd",
               "durationSeconds", "aggregates.casesPassed", "aggregates.casesTotal", "cases[].name",
               "cases[].aggregates.score", "cases[].aggregates.delta", "cases[].arms.with[]",
               "cases[].arms.without[]", "run.error", "run.aborted", "run.skippedPaidGraders", "grader.passed",
               "grader.scored", "grader.weight", "run.score", "run.graders", "plugin.json")

# What each object's documented members become; everything else is kept under VENDOR_NS.
SUITE_KNOWN = ("schemaVersion", "claudeVersion", "partial", "partialReason", "costUsd", "durationSeconds", "cases")
SUITE_AGG_KNOWN = ("overallScore", "casesPassed", "casesTotal", "meanDelta")
CASE_KNOWN = ("name",)
CASE_AGG_KNOWN = ("score", "delta")
RUN_KNOWN = ("error", "skippedPaidGraders", "score")
ABORTED_KNOWN = ("server", "tool")
PARTIAL_REASONS = ("cost_ceiling", "interrupted", "auth_failed")   # documented (#json-result)

WITHHELD_NO_DELTA = ("both arms ran but the file gives no delta, which the docs omit when the arms are not "
                     "comparable")
WITHHELD_ONE_ARM = ("one arm: the docs run a case with one arm under --ablation none, a resumed transcript, or no "
                    "plugin found, so it has no baseline")
WITHHELD_NO_WITH = "no with runs"


class FormatError(ValueError):
    """A file plugin eval did not write, a result version this reader does not know, or a reading that does not
    reproduce the file's own case scores, named."""


@dataclass
class Result:
    rows: list[dict[str, Any]]
    format: str
    skipped: list[dict[str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Run:
    arm: str
    j: int                      # 1-based, in file order
    raw: dict[str, Any]
    grade: float | None         # the file's grade; None when ungraded (a withheld case only)
    grade_from: str | None      # score | graders | aborted
    skipped: bool               # skippedPaidGraders: unscored


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _g(x: float) -> str:
    return f"{x:.6g}"


def _counted(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------- the input
def result_file(path: str | Path) -> Path:
    """`path` itself, or the `aggregate-result.json` in the folder `path` names."""
    p = Path(path).expanduser()
    if p.is_dir():
        f = p / RESULT_FILE
        if not f.is_file():
            raise FormatError(f"{p}: no {RESULT_FILE} in this folder (plugin eval writes it to <eval dir>/results/"
                              f"<timestamp>/{RESULT_FILE}, and --json writes the same document)")
        return f
    return p


def _load(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FormatError(f"{path}: cannot read: {exc.strerror or exc}") from None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{path}: not JSON ({exc}); a plugin eval result is one JSON object") from None
    if not isinstance(doc, dict):
        raise FormatError(f"{path}: the result must be an object, not {type(doc).__name__}")
    version = doc.get("schemaVersion")
    if "schemaVersion" not in doc:
        raise FormatError(f"{path}: no schemaVersion, so not a claude plugin eval result (this reader knows "
                          f"schemaVersion {', '.join(map(str, SCHEMA_VERSIONS))})")
    if isinstance(version, bool) or not isinstance(version, int) or version not in SCHEMA_VERSIONS:
        raise FormatError(f"schemaVersion {version!r} is not known (this reader knows "
                          f"{', '.join(map(str, SCHEMA_VERSIONS))})")
    if not isinstance(doc.get("cases"), list):
        raise FormatError(f"{path}: no cases list; a plugin eval result lists its cases under cases")
    return raw, doc


def _plugin(path: Path, given: str | None, notes: list[str]) -> dict[str, Any] | None:
    """The plugin's name and version: `--plugin NAME`, else its manifest, found only when the result sits where
    plugin eval writes it (`<plugin>/<eval dir>/results/<timestamp>/aggregate-result.json`), at the plugin
    root as `plugin.json` or `.claude-plugin/plugin.json` (https://code.claude.com/docs/en/plugin-evals)."""
    if given is not None:
        return {"name": given, "version": None, "from": "--plugin"}
    folder = path.resolve().parent
    if folder.parent.name == "results":
        start = folder.parent.parent.parent   # above the eval dir, which may itself be nested
        for root in [start, *list(start.parents)[:3]]:
            for rel in (".claude-plugin/plugin.json", "plugin.json"):
                manifest = root / rel
                if not manifest.is_file():
                    continue
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    notes.append(f"{rel} beside the results does not parse; pass --plugin NAME")
                    return None
                name = data.get("name") if isinstance(data, dict) else None
                if isinstance(name, str) and name.strip() and "\n" not in name:
                    version = data.get("version")
                    return {"name": name.strip(), "version": version if isinstance(version, str) else None,
                            "from": rel}
                notes.append(f"{rel} beside the results names no plugin; pass --plugin NAME")
                return None
    notes.append("no plugin.json found where plugin eval keeps its results, and no --plugin NAME: the arms have "
                 "no harness record and compare by their labels only; pass --plugin NAME to record the plugin")
    return None


def _record(arm: str, plugin: dict[str, Any] | None) -> dict[str, Any] | None:
    """The arm's harness record. No tree is in the file, so the digest is this reader's own stable id
    (spec/harness.md section 2); the factor names the plugin, not its version, so evidence pools across
    versions (section 4)."""
    if plugin is None:
        return None
    key = f"plugin-eval\0{arm}\0{plugin['name']}\0{plugin.get('version') or ''}"
    return {"schema": HARNESS_SCHEMA, "digest": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "parent": None, "source": "plugin-eval", "layouts": [],
            "factors": [f"plugin:{plugin['name']}"] if arm == "with" else []}


# ---------------------------------------------------------------- one run's grade
def _grader_lists(run: dict[str, Any]) -> list[str]:
    """The run's members that hold a grader list: a list of objects, any of which has `passed`."""
    return sorted(k for k, v in run.items() if isinstance(v, list)
                  and any(isinstance(i, dict) and "passed" in i for i in v))


def _fraction(items: list[Any], where: str, key: str) -> float | None:
    """The weighted fraction passed over the graders that count; None when none counts."""
    total = passed = 0.0
    for i, g in enumerate(items, 1):
        at = f"{where}: grader {i} in {key}"
        if not isinstance(g, dict):
            raise FormatError(f"{at}: an object, not {type(g).__name__}")
        if not isinstance(g.get("passed"), bool):
            raise FormatError(f"{at}: passed must be true or false, not {g.get('passed')!r}")
        scored = g.get("scored", True)
        if not isinstance(scored, bool):
            raise FormatError(f"{at}: scored must be true or false, not {scored!r}")
        weight = g.get("weight", 1.0)
        if not _number(weight) or weight < 0:
            raise FormatError(f"{at}: weight must be a number at or above 0, not {weight!r}")
        if scored:
            total += float(weight)
            passed += float(weight) if g["passed"] else 0.0
    return passed / total if total > 0 else None


def _grade(run: Any, where: str, *, withheld: bool) -> tuple[float | None, str | None, bool]:
    """(grade, where it came from, skippedPaidGraders) for one run, by the provisional layout
    (docs/plugin-eval.md). Refuses what admits two readings or none."""
    if not isinstance(run, dict):
        raise FormatError(f"{where}: a run must be an object, not {type(run).__name__}")
    skipped = run.get("skippedPaidGraders", False)
    if not isinstance(skipped, bool):
        raise FormatError(f"{where}: skippedPaidGraders must be true or false, not {skipped!r}")
    aborted = run.get("aborted")
    if aborted is not None and not isinstance(aborted, dict):
        raise FormatError(f"{where}: aborted must be an object or null, not {type(aborted).__name__}")
    error = run.get("error")
    if error is not None and not isinstance(error, str):
        raise FormatError(f"{where}: error must be a string or null, not {type(error).__name__}")
    has_score = "score" in run
    score = run.get("score")
    if has_score and score is None:
        if not skipped:
            raise FormatError(f"{where}: score is null on a run whose paid graders were not skipped")
        has_score = False
    elif has_score and not (_number(score) and 0.0 <= score <= 1.0):
        raise FormatError(f"{where}: score must be a number from 0 to 1, not {score!r}")
    lists = _grader_lists(run)
    if len(lists) > 1:
        raise FormatError(f"{where}: two grader lists ({', '.join(lists)}); this reader reads one, and will not "
                          f"guess which")
    frac = _fraction(run[lists[0]], where, lists[0]) if lists else None
    if aborted is not None:   # documented: the run scores 0 and error stays null
        return 0.0, "aborted", skipped
    if has_score:
        if lists and frac is not None and abs(frac - float(score)) > TOL:
            raise FormatError(f"{where}: score {_g(score)} but its grader list ({lists[0]}) gives {_g(frac)}; "
                              f"this reader will not pick one")
        return float(score), "score", skipped
    if lists:
        if frac is None:
            raise FormatError(f"{where}: no grader in {lists[0]} counts (every one has scored false or weight 0), "
                              f"and the run has no score")
        return frac, "graders", skipped
    if withheld:   # listed with no grade, never 0
        return None, None, skipped
    raise FormatError(f"{where}: no score and no grader list; the file gives this run no grade, and this reader "
                      f"supplies none (an API error ends a run with no grader results; re-run it once the error "
                      f"is resolved)")


# ---------------------------------------------------------------- the case check
def _means(runs: list[_Run]) -> list[float | None]:
    """The arm's mean grade under both readings of a skippedPaidGraders run: in the mean, and left out."""
    out: list[float | None] = []
    for keep_skipped in (True, False):
        use = [r.grade for r in runs if keep_skipped or not r.skipped]
        out.append(sum(use) / len(use) if use and None not in use else None)
    return out


def _check_case(name: str, agg: dict[str, Any], runs: dict[str, list[_Run]], version: str | None) -> None:
    """The file's own numbers must come back: the mean `with` grade is `aggregates.score`, and with minus
    without is `aggregates.delta`, under one reading of the skipped runs."""
    if any(r.grade is None for r in runs["with"]):
        return   # an ungraded run (withheld case only): nothing to check against
    score, delta = agg.get("score"), agg.get("delta")
    if score is not None and not _number(score):
        raise FormatError(f"case {name}: aggregates.score must be a number, not {score!r}")
    if delta is not None and not _number(delta):
        raise FormatError(f"case {name}: aggregates.delta must be a number, not {delta!r}")
    with_m, without_m = _means(runs["with"]), _means(runs["without"])
    for i in range(2):
        if with_m[i] is None:
            ok_score = score is None
        else:
            ok_score = score is not None and abs(with_m[i] - score) <= TOL
        ok_delta = delta is None or (with_m[i] is not None and without_m[i] is not None
                                     and abs(with_m[i] - without_m[i] - delta) <= TOL)
        if ok_score and ok_delta:
            return
    shown = ", ".join(f"{k} {_g(v)}" for k, v in (("aggregates.score", score), ("aggregates.delta", delta))
                      if v is not None)
    mine = (f"with mean {_g(with_m[0]) if with_m[0] is not None else 'none'}"
            + (f", without mean {_g(without_m[0])}" if without_m[0] is not None else ""))
    raise FormatError(f"case {name}: this reader's reading of the run grades does not match the file's case scores "
                      f"({mine}; the file: {shown}; claudeVersion {version or 'not given'}). The result's layout "
                      f"may not be the one this reader knows ({DOCS}); nothing is imported from a misreading")


# ---------------------------------------------------------------- kept fields and text
def _strip(value: Any, path: str, keep_text: bool, dropped: set[str]) -> Any:
    """`value` with every string dropped (its path recorded), unless `keep_text`."""
    if isinstance(value, str):
        if keep_text:
            return value
        dropped.add(path)
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            kept = _strip(v, f"{path}.{k}", keep_text, dropped)
            if kept is not None or not isinstance(v, str):
                out[k] = kept
        return out
    if isinstance(value, list):
        kept = [(v, _strip(v, f"{path}[]", keep_text, dropped)) for v in value]
        return [x for v, x in kept if x is not None or not isinstance(v, str)]
    return value


def _unknown(obj: dict[str, Any], known: tuple[str, ...], agg_known: tuple[str, ...] = ()) -> dict[str, Any]:
    out = {k: v for k, v in obj.items() if k not in known and k != "aggregates"}
    agg = obj.get("aggregates")
    if isinstance(agg, dict):
        rest = {k: v for k, v in agg.items() if k not in agg_known}
        if rest:
            out["aggregates"] = rest
    elif "aggregates" in obj:
        out["aggregates"] = agg
    return out


# ---------------------------------------------------------------- the cost file
def _costs(path: Path, cases: dict[str, dict[str, list[_Run]]], written: set[str],
           notes: list[str]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], int]:
    """{(case, arm, run): {usd?, tokens?, model?}} from a user's own JSON lines file, and its line count.
    Every line must name exactly one run of the result; a line for a withheld case is counted and unused."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FormatError(f"--cost-from {path}: cannot read: {exc}") from None
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    lines = unused = 0
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        lines += 1
        at = f"{path}:{n}"
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormatError(f"{at}: not JSON ({exc})") from None
        if not isinstance(item, dict):
            raise FormatError(f"{at}: a line is one object {{case, arm, run, usd and/or tokens, model}}")
        case, arm, j = item.get("case"), item.get("arm"), item.get("run")
        if not isinstance(case, str) or case not in cases:
            raise FormatError(f"{at}: case {case!r} is not in the result")
        if arm not in ARMS:
            raise FormatError(f"{at}: arm {arm!r} is not with or without")
        if isinstance(j, bool) or not isinstance(j, int) or j < 1:
            raise FormatError(f"{at}: run {j!r} must be a whole number from 1")
        have = len(cases[case][arm])
        if j > have:
            raise FormatError(f"{at}: run {j}: case {case} has {_counted(have, f'{arm} run')}")
        usd, tokens, model = item.get("usd"), item.get("tokens"), item.get("model")
        if usd is None and tokens is None:
            raise FormatError(f"{at}: give usd or tokens for case {case}, {arm} run {j}")
        if usd is not None and not (_number(usd) and usd >= 0):
            raise FormatError(f"{at}: usd must be a number at or above 0, not {usd!r}")
        if tokens is not None and not isinstance(tokens, dict):
            raise FormatError(f"{at}: tokens must be an object {{input, cache_read, cache_write, output}}")
        if model is not None and not (isinstance(model, str) and model.strip()):
            raise FormatError(f"{at}: model must be a model name")
        key = (case, arm, j)
        if key in out:
            raise FormatError(f"{at}: case {case}, {arm} run {j} already has a cost line")
        out[key] = {k: v for k, v in (("usd", usd), ("tokens", tokens), ("model", model)) if v is not None}
        unused += case not in written
    if unused:
        notes.append(f"{_counted(unused, 'cost line')} in --cost-from name runs of withheld cases and are not used")
    return out, lines


# ---------------------------------------------------------------- the reader
def read(path: str | Path, *, experiment: str | None = None, cost_from: str | Path | None = None,
         accept_at: float | None = None, outcome: str = "accept", model: str | None = None,
         plugin: str | None = None, keep_text: bool = False) -> Result:
    """The rows of one plugin eval result (docs/plugin-eval.md). Raises FormatError naming what is wrong."""
    if outcome not in ("accept", "score"):
        raise FormatError(f"outcome {outcome!r}: accept or score")
    file = result_file(path)
    raw, doc = _load(file)
    notes: list[str] = []
    version = doc.get("claudeVersion") if isinstance(doc.get("claudeVersion"), str) else None
    exp = experiment or "plugin-eval-" + hashlib.sha256(raw).hexdigest()[:12]
    threshold = float(accept_at) if accept_at is not None else DEFAULT_THRESHOLD
    plug = _plugin(file, plugin, notes)

    # the cases, their arms, and which are compared
    names: list[str] = []
    for i, c in enumerate(doc["cases"], 1):
        if not isinstance(c, dict):
            raise FormatError(f"case {i}: an object, not {type(c).__name__}")
        name = c.get("name")
        if not (isinstance(name, str) and name.strip() and "\n" not in name):
            raise FormatError(f"case {i}: name must be a one-line string, not {name!r}")
        if name in names:
            raise FormatError(f"two cases are named {name}; a case name is the task id, so each must be unique")
        names.append(name)
        if not isinstance(c.get("arms"), dict):
            raise FormatError(f"case {name}: no arms object (arms.with[] and arms.without[])")
        if not isinstance(c.get("aggregates", {}), dict):
            raise FormatError(f"case {name}: aggregates must be an object")
        for arm in ARMS:
            if c["arms"].get(arm) is not None and not isinstance(c["arms"][arm], list):
                raise FormatError(f"case {name}: arms.{arm} must be a list")
    cases = {c["name"]: c for c in doc["cases"]}
    two_arm = any(c["arms"].get("without") for c in cases.values())
    why_held: dict[str, str] = {}
    if two_arm:
        for name, c in cases.items():
            if not c["arms"].get("with"):
                why_held[name] = WITHHELD_NO_WITH
            elif not c["arms"].get("without"):
                why_held[name] = WITHHELD_ONE_ARM
            elif (c.get("aggregates") or {}).get("delta") is None:
                why_held[name] = WITHHELD_NO_DELTA

    # every run's grade, then each case's check against its own numbers
    graded: dict[str, dict[str, list[_Run]]] = {}
    for name, c in cases.items():
        graded[name] = {}
        for arm in ARMS:
            runs = []
            for j, r in enumerate(c["arms"].get(arm) or [], 1):
                grade, src, skipped = _grade(r, f"case {name}, {arm} run {j}", withheld=name in why_held)
                runs.append(_Run(arm, j, r, grade, src, skipped))
            graded[name][arm] = runs
        _check_case(name, c.get("aggregates") or {}, graded[name], version)

    compared = [n for n in names if n not in why_held and graded[n]["with"]]
    if two_arm and not compared:
        held = [n for n in names if why_held.get(n) == WITHHELD_NO_DELTA]
        detail = (f"{_counted(len(held), 'case')} with both arms and no delta ({', '.join(held)})" if held
                  else "no case has both arms")
        raise FormatError(f"no case is comparable: {detail}; the docs omit delta when the case ran one arm or the "
                          f"arms are not comparable")

    costs, cost_lines = ({}, 0)
    if cost_from is not None:
        costs, cost_lines = _costs(Path(cost_from).expanduser(), graded, set(compared), notes)

    # the suite's own numbers, kept on every row
    agg = doc.get("aggregates") if isinstance(doc.get("aggregates"), dict) else {}
    partial = doc.get("partial") is True
    reason = doc.get("partialReason") if isinstance(doc.get("partialReason"), str) else None
    cost_usd = doc.get("costUsd") if _number(doc.get("costUsd")) else None
    suite = {"schema_version": doc["schemaVersion"], "claude_version": version, "cost_usd": cost_usd,
             "duration_seconds": doc.get("durationSeconds") if _number(doc.get("durationSeconds")) else None,
             "partial": partial,
             "partial_reason": reason if reason in PARTIAL_REASONS else None,   # any other reason is free text
             "cases_passed": agg.get("casesPassed") if _number(agg.get("casesPassed")) else None,
             "cases_total": agg.get("casesTotal") if _number(agg.get("casesTotal")) else None,
             "threshold": threshold, "threshold_from": "--accept-at" if accept_at is not None else "default"}
    dropped: set[str] = set()
    vendor_suite = _unknown(doc, SUITE_KNOWN, SUITE_AGG_KNOWN)
    if reason is not None and reason not in PARTIAL_REASONS:
        vendor_suite["partialReason"] = reason
    vendor_suite = _strip(vendor_suite, "suite", keep_text, dropped)

    rows: list[dict[str, Any]] = []
    records = {arm: _record(arm, plug) for arm in ARMS}
    errors = {arm: 0 for arm in ARMS}
    unscored = 0
    written = compared if two_arm else [n for n in names if graded[n]["with"]]
    for name in written:
        c = cases[name]
        cagg = c.get("aggregates") or {}
        ours_case = {"score": cagg.get("score"), "delta": cagg.get("delta")}
        vendor_case = _strip(_unknown({k: v for k, v in c.items() if k != "arms"}, CASE_KNOWN, CASE_AGG_KNOWN),
                             "case", keep_text, dropped)
        arms_rest = {k: v for k, v in c["arms"].items() if k not in ARMS}
        if arms_rest:
            vendor_case["arms"] = _strip(arms_rest, "case.arms", keep_text, dropped)
        for arm in ARMS:
            for r in graded[name][arm]:
                run = r.raw
                aborted = run.get("aborted")
                ours_run = {"index": r.j, "grade": r.grade, "grade_from": r.grade_from,
                            "errored": run.get("error") is not None,
                            "aborted": ({k: aborted.get(k) for k in ABORTED_KNOWN} if aborted is not None else None),
                            "skipped_paid_graders": r.skipped}
                vendor_run = {k: v for k, v in run.items() if k not in RUN_KNOWN and k != "aborted"}
                if run.get("error") is not None:   # free text: kept with the vendor's fields, so --keep-text rules it
                    vendor_run["error"] = run["error"]
                if aborted is not None:
                    rest = {k: v for k, v in aborted.items() if k not in ABORTED_KNOWN}
                    if rest:
                        vendor_run["aborted"] = rest
                vendor_run = _strip(vendor_run, "run", keep_text, dropped)
                errors[arm] += run.get("error") is not None
                if r.skipped or r.grade is None:
                    out = None
                    unscored += 1
                elif outcome == "accept":
                    out = {"accepted": r.grade >= threshold - 1e-9, "by": TOOL, "grade": r.grade,
                           "threshold": threshold}
                else:
                    out = {"name": SCORE_NAME, "score": r.grade, "scale": "fraction", "judge": TOOL}
                line = costs.get((name, arm, r.j))
                cost = None
                if line:
                    cost = {k: line[k] for k in ("usd", "tokens") if k in line}
                    if "usd" in cost:
                        cost["usd_source"] = "cost-from"
                row = {"schema": ROWS_SCHEMA, "id": f"plugin-eval:{exp}:{name}:{arm}:{r.j}", "task_id": name,
                       "arm": arm, "pair": f"{name}#{r.j}", "experiment": exp, "status": "done",
                       "harness": records[arm], "agent": "claude-code",
                       "model": (line or {}).get("model") or model, "effort": None, "outcome": out, "cost": cost,
                       "started_at": None, "ended_at": None,
                       "tool": {"name": TOOL, "version": version, "run_id": f"{name}/{arm}/{r.j}", "task_id": name},
                       "ext": {NS: {"suite": dict(suite), "case": dict(ours_case), "run": ours_run},
                               VENDOR_NS: {"suite": vendor_suite, "case": vendor_case, "run": vendor_run}}}
                rows.append(row)

    # notes and the summary
    runs = {arm: sum(len(graded[n][arm]) for n in written) for arm in ARMS}
    pairs = sum(min(len(graded[n]["with"]), len(graded[n]["without"])) for n in written)
    for n in written:
        w, wo = len(graded[n]["with"]), len(graded[n]["without"])
        if two_arm and w != wo:
            notes.append(f"case {n} has {_counted(w, 'with run')} and {wo} without: "
                         f"{_counted(abs(w - wo), 'extra run')}, unpaired, in the rows as lone runs")
    if not two_arm:
        notes.append("no case has a without arm (one arm, as under --ablation none): the with runs are imported as "
                     "records, and there is nothing to compare")
    held = [{"case": n, "why": why_held[n],
             "runs": {arm: [None if r.skipped else r.grade for r in graded[n][arm]] for arm in ARMS}}
            for n in names if n in why_held]
    if held:
        notes.append("withheld from the comparison: " + "; ".join(f"{h['case']} ({h['why']})" for h in held)
                     + f"; no row is written for {'it' if len(held) == 1 else 'them'}, and the run grades are in "
                     f"the summary's withheld list")
    if two_arm and pairs > len(compared):
        shared = (f"{_counted(pairs, 'pair')} from {_counted(len(compared), 'compared case')}: the runs of one case "
                  f"share its task")
        if outcome == "score":
            notes.append(f"{shared}, so check-change from-runs compares the cases, one pair each with each arm's mean "
                         f"grade, for an answer about new cases; gate pairs the runs, so its answer is about these "
                         f"cases; to learn about tasks like these, add cases rather than runs ({DOCS})")
        else:
            notes.append(f"{shared}, and acceptance rows are compared run by run: the answer is about these cases, "
                         f"and an adopt from a few cases with several runs each carries to new cases less well when "
                         f"the plugin helps some cases and hurts others; to learn about tasks like these, add cases "
                         f"rather than runs ({DOCS})")
    if sum(errors.values()):
        notes.append(f"{_counted(sum(errors.values()), 'run')} ended with an error ({errors['with']} with, "
                     f"{errors['without']} without); each keeps the grade the file gives it. Since Claude Code "
                     f"2.1.299 a run that ends on an API error scores 0 with no grader results, so an infrastructure "
                     f"error counts against the arm it hit: re-run those cases")
    if partial:
        notes.append(f"the suite did not finish (partial: {reason or 'no reason given'}): the runs "
                     f"it recorded are imported, and runs it never made are not rows")
    reach = sum(1 for c in cases.values()
                if _number((c.get("aggregates") or {}).get("score")) and c["aggregates"]["score"] >= threshold - 1e-9)
    if suite["cases_passed"] is not None and int(suite["cases_passed"]) != reach:
        if accept_at is None:
            notes.append(f"casesPassed is {int(suite['cases_passed'])}, but {reach} of the cases reach the default "
                         f"threshold {DEFAULT_THRESHOLD:g}: the suite likely ran with another --threshold; pass "
                         f"--accept-at X to accept runs at it")
        else:
            notes.append(f"casesPassed is {int(suite['cases_passed'])}, but {reach} of the cases reach --accept-at "
                         f"{threshold:g}")
    if model is None and not any(r["model"] for r in rows):
        notes.append("the result does not record the model; pass --model M to label the runs")
    notes.append(f"the run layout read here is provisional ({DOCS}): each case's reading is checked against its "
                 f"score and delta, and real output is not yet verified")

    cost_summary = None
    if cost_from is not None:
        with_cost = sum(1 for r in rows if r["cost"])
        cost_summary = {"lines": cost_lines, "runs_with_cost": with_cost, "runs_without": len(rows) - with_cost}
    summary = {"plugin": plug, "cases": len(names), "compared": len(compared) if two_arm else 0, "runs": runs,
               "pairs": pairs, "unscored": unscored, "errors": errors, "experiment": exp,
               "threshold": {"value": threshold, "from": suite["threshold_from"]}, "outcome": outcome,
               "claude_version": version, "partial": (reason or "true") if partial else None,
               "suite_cost_usd": cost_usd, "cost_from": cost_summary, "dropped_text": sorted(dropped),
               "withheld": held}
    return Result(rows=rows, format=FORMAT, counts={"cases": len(names), "rows": len(rows), "withheld": len(held)},
                  experiments=[{"experiment": exp, "runs": len(rows)}] if rows else [], notes=notes, summary=summary)
