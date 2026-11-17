"""OCP signals, preferences and harness-runs rows to and from the score tables.

Not a command: `scores.py` calls it. The only module of `loopmath.eval` that may import `loopmath.ocp` and
`loopmath.harness_runs`.

The mapping, both ways:

- A reading is a score signal. `id` is the reading id, `name` the component, `source.ref` the judge, and
  `ext["dev.loopmath.eval"]` holds `call`, `step`, `rubric`, `replaces`, `assignment`, `item` and `of`.
  A signal's value is a number, so a level that is a word is written as `level` beside them, with the
  value null. `cost_usd` and the members of the reading's own `ext` are kept there too.
- A pairwise judgement is a preference on a slate of runs of one input: `winner` is a run id or `tie`,
  `members` is `[a, b]`, the judge id sits in `judge.ext["dev.loopmath.eval"].id`, and
  `ext["dev.loopmath.eval"]` holds `component`, `step`, `call`, `shown_first`, `rubric`, `replaces` and
  `assignment`.
- What an OCP record holds beyond these columns (a tier, a unit, an attempt, another namespace) is kept on
  the study row under `ext["dev.loopmath.eval.ocp"]` and written back on export.

Records made by other tools have no `dev.loopmath.eval` member. For those, per (run, judge, name) the
newest score signal is the reading and each earlier one is a reading it corrects. A signal id of another
tool is unique in its run document, not in a study, so the reading's id and call are `scoped_id(run id,
signal id)`: two documents that both hold a signal `s1` give two readings. A preference of another tool
is named within its slate the same way, `scoped_id(slate, preference id)`, so two slates that both hold
a preference `f1` give two pairs, and the copies of one preference in its members' files give one. The
record's own id is kept and written back on export, where the eval member's `id` names the reading or
pair. A preference is read on the overall component, or the one the caller names, with `shown_first`
unknown.

Nothing here writes a file. `read_documents` and `read_rows` return rows for `scores.py` to validate and
merge; `export` returns what `eval scores export` prints.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...harness_runs.rows import read_numbered, run_id_for, stage_piece
from ...ocp.conformance import SIMULATE_KEY, validate_doc
from .. import contracts as C

NS = "dev.loopmath.eval"
KEPT = "dev.loopmath.eval.ocp"
READING_MEMBERS = ("call", "step", "rubric", "replaces", "assignment", "item", "of")
PAIR_MEMBERS = ("component", "step", "call", "shown_first", "rubric", "replaces", "assignment")
ALSO = ("level", "cost_usd", "observed_at", "ext")  # written beside them when the row needs it
JUDGE_SOURCES = ("judge", "user")
_OCP_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_WORD = re.compile(r"[A-Za-z]+|\d+")


@dataclass
class Batch:
    """What one source gave: raw rows for the study tables, and every record that was not used.

    `runs`, `scores` and `pairs` are not validated yet. `source` names where a row came from, by table and
    id, for refusal messages. Each entry of `not_used` is `{"source", "kind", "id", "reason"}`."""

    runs: list[dict] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)
    not_used: list[dict] = field(default_factory=list)
    source: dict[tuple[str, str], str] = field(default_factory=dict)


def run(args: argparse.Namespace) -> int:
    """The adapter is not a command. `eval scores` is the command that uses it."""
    from ..scores import run as scores_run

    return scores_run(args)


def family_of(model_id: str) -> str:
    """A family label for a model id that states none: its first word, lower case (`model-large` gives `model`).

    A guess, used only when a record names no family. State `model.family` in the record to set it."""
    words = _WORD.findall(model_id)
    return words[0].lower() if words else "unknown"


def default_slate(input_id: str) -> str:
    """The slate id `export` gives the pairs of one input that came with no slate of their own."""
    return f"eval.{input_id}"[:200]


def _obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _instant(value: Any) -> float:
    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError):
        return 0.0


def _total(costs: Iterable[Any]) -> float | None:
    """The sum of the costs that are numbers, or None when there is none or the sum is not a finite number."""
    numbers = [x for x in costs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    try:
        total = float(sum(numbers))
    except OverflowError:
        return None
    return total if numbers and math.isfinite(total) else None


def _ident(text: str) -> str:
    return "".join("_" if ch.isspace() or ch in "/@" else ch for ch in text)


# ---------------------------------------------------------------------------------------------------
# OCP run documents to rows

def _step(step: str, model: Any, effort: Any, ext: Any = None) -> dict:
    ref = model if isinstance(model, dict) else {"id": model}
    model_id = ref.get("id") or ref.get("raw") or "unknown"
    family = ref.get("family") if isinstance(ref.get("family"), str) and ref.get("family") else family_of(model_id)
    out: dict = {"step": step, "model": {"id": model_id, "family": _ident(family)}}
    if isinstance(effort, str) and effort:
        out["effort"] = effort
    variant = _obj(_obj(ext).get(NS)).get("variant")
    if isinstance(variant, str) and variant:
        out["variant"] = variant
    return out


def run_row(doc: dict, file: str) -> dict:
    """The study's run row of one OCP run document.

    `input` is `run.task.id`. `arm` is `run.flow.arm`, else `run.labels.arm`, else the configuration id.
    `steps` are the workflow's pieces in order with their settings; without a configuration, the last
    attempt of each node. `cost_usd` is the sum of the attempts' `cost.usd`."""
    run = _obj(doc.get("run"))
    task, flow, config = _obj(run.get("task")), _obj(run.get("flow")), _obj(run.get("configuration"))
    if not isinstance(run.get("id"), str) or not isinstance(task.get("id"), str):
        raise C.ContractError(file, "run", "the document has no run.id or no run.task.id",
                              "give an OCP run document at version 0.3 or later, with its task", "no_run")
    arm = flow.get("arm") or _obj(run.get("labels")).get("arm") or config.get("id")
    if not isinstance(arm, str) or not arm:
        raise C.ContractError(file, "run", f'run "{run["id"]}" names no arm',
                              "state run.flow.arm, run.labels.arm or a configuration id", "no_arm")
    out: dict = {"run": run["id"], "input": task["id"], "arm": arm}
    if isinstance(flow.get("split"), str) and flow["split"]:
        out["split"] = flow["split"]
    settings = _obj(config.get("settings"))
    pieces = [p.get("id") for p in _obj(config.get("workflow")).get("pieces") or [] if isinstance(p, dict)]
    order = [p for p in pieces if p in settings] + [p for p in settings if p not in pieces]
    steps = [_step(_ident(p), _obj(settings[p]).get("model") or {}, _obj(settings[p]).get("effort"),
                   _obj(settings[p]).get("ext")) for p in order if isinstance(settings[p], dict)]
    attempts = [a for a in doc.get("attempts") or [] if isinstance(a, dict)]
    if not steps:
        last: dict[str, dict] = {}
        for attempt in attempts:
            where = attempt.get("vertex") or attempt.get("node")
            if isinstance(where, str) and where:
                last.pop(where, None)
                last[where] = attempt
        steps = [_step(_ident(where), a.get("model") or {}, a.get("effort"), a.get("ext")) for where, a in last.items()]
    if not steps:
        raise C.ContractError(file, "run", f'run "{run["id"]}" has no step settings',
                              "give run.configuration.settings, or attempts with their models", "no_steps")
    out["steps"] = steps
    cost = _total(_obj(a.get("cost")).get("usd") for a in attempts)
    if cost is not None:
        out["cost_usd"] = cost
    out["simulated"] = isinstance(_obj(run.get("ext")).get(SIMULATE_KEY), dict)
    slate = run.get("slate")
    if isinstance(slate, dict) and (slate.get("id") != default_slate(task["id"]) or set(slate) - {"id", "members"}):
        out["ext"] = {KEPT: {"slate": copy.deepcopy(slate)}}  # a slate `export` would not write by itself
    return out


def _rest(record: dict, columns: Iterable[str], known: Iterable[str]) -> dict:
    """What an OCP record holds beyond the study row's columns, for `ext[KEPT]`."""
    rest = {k: copy.deepcopy(v) for k, v in record.items() if k not in columns and k != "ext"}
    if rest.get("tier") == "reported":
        del rest["tier"]
    ext = {k: copy.deepcopy(v) for k, v in _obj(record.get("ext")).items() if k != NS}
    if ext:
        rest["ext"] = ext
    more = {k: copy.deepcopy(v) for k, v in _obj(_obj(record.get("ext")).get(NS)).items()
            if k not in known and k not in ALSO}
    if more:
        rest["eval"] = more
    return rest


def _common(row: dict, e: dict, names: Iterable[str], rest: dict) -> dict:
    for name in names:
        if e.get(name) is not None:
            row[name] = e[name]
    ext = dict(_obj(e.get("ext")))
    if rest:
        ext[KEPT] = rest
    if ext:
        row["ext"] = ext
    return row


def _escape(text: str, also: str = "") -> str:
    """`text` with `%`, white space, `/`, `@` and the characters of `also` written as `%XX`, one per UTF-8 byte."""
    return "".join("".join(f"%{byte:02X}" for byte in ch.encode("utf-8")) if ch.isspace() or ch in "%/@" + also
                   else ch for ch in text)


def scoped_id(scope: str, record_id: str) -> str:
    """The study-wide id of a record that another tool named within a scope: a signal within its run, a
    preference within its slate.

    It is `<scope>.<record id>` with `%`, white space, `/` and `@` written as `%XX` (per UTF-8 byte) in both
    parts, and `.` as `%2E` in the scope. The first `.` left in the id ends the scope and `%` always starts
    an escape, so two different pairs never give one id: run `a.b` with signal `c` gives `a%2Eb.c`, run `a`
    with signal `b.c` gives `a.b.c`. Ids of letters, digits, `-` and `_` read as they are."""
    return f"{_escape(scope, '.')}.{_escape(record_id)}"


def reading_row(run_id: str, signal: dict, rubric_ref: str | None = None, *, scope: bool = False) -> dict:
    """The study's reading of one score signal. Not validated: `contracts.validate_score` does that.

    The reading's id is the eval member's `id`, else the signal's id. With `scope`, a signal that has no
    eval member is named within its run: its reading id and call are `scoped_id(run, signal id)`. When
    the reading's id is not the signal's, the signal's id is kept for `signal` to write back."""
    stated = _obj(signal.get("ext")).get(NS)
    e = _obj(stated)
    source = _obj(signal.get("source"))
    given = signal.get("id")
    ident = e.get("id") or given
    if scope and not isinstance(stated, dict) and isinstance(given, str) and given:
        ident = scoped_id(run_id, given)
    row = {"id": ident, "run": run_id, "step": e.get("step") or C.FINAL, "component": signal.get("name"),
           "judge": source.get("ref"), "call": e.get("call") or ident,
           "value": e["level"] if e.get("level") is not None else signal.get("value"),
           "rubric": e.get("rubric") or rubric_ref, "observed_at": e.get("observed_at") or signal.get("observed_at")}
    rest = _rest(signal, ("id", "kind", "name", "value", "observed_at", "source"), (*READING_MEMBERS, "id"))
    if ident != given:
        rest["id"] = given
    kept_source = {k: copy.deepcopy(v) for k, v in source.items() if k != "ref"}
    if kept_source not in ({"kind": "judge"}, {"kind": "user"}):  # which of the two: the judges file says
        rest["source"] = kept_source
    return _common(row, e, ("item", "of", "replaces", "assignment", "cost_usd"), rest)


def _judge_id(judge: dict) -> str:
    stated = _obj(_obj(judge.get("ext")).get(NS)).get("id")
    if isinstance(stated, str) and stated:
        return stated
    return _ident(str(judge.get("model") or judge.get("kind") or "referee"))


def pair_row(preference: dict, members: list, inputs: Mapping[str, str], rubric_ref: str | None = None,
             component: str | None = None, observed_at: str | None = None, *, scope: bool = False) -> dict:
    """The study's pair of one preference between the two runs `members`. Not validated.

    The pair's id is the eval member's `id`, else the preference's id. With `scope`, a preference that has
    no eval member is named within its slate: its pair id and call are `scoped_id(slate, preference id)`.
    When the pair's id is not the preference's, the preference's id is kept for `preference` to write back."""
    stated = _obj(preference.get("ext")).get(NS)
    e = _obj(stated)
    a, b = members
    judge = _obj(preference.get("judge"))
    winner = preference.get("winner")
    given, slate = preference.get("id"), preference.get("slate")
    ident = e.get("id") or given
    if scope and not isinstance(stated, dict) and isinstance(given, str) and given and isinstance(slate, str):
        ident = scoped_id(slate, given)
    row = {"id": ident, "a": a, "b": b, "step": e.get("step") or C.FINAL,
           "component": e.get("component") or component, "judge": _judge_id(judge),
           "call": e.get("call") or ident, "shown_first": e.get("shown_first"),
           "verdict": "a" if winner == a else "b" if winner == b else winner,
           "rubric": e.get("rubric") or rubric_ref,
           "observed_at": e.get("observed_at") or preference.get("observed_at") or observed_at}
    rest = _rest(preference, ("id", "winner", "members", "observed_at", "judge", "slate"), (*PAIR_MEMBERS, "id"))
    if ident != given:
        rest["id"] = given
    kept_judge = {k: copy.deepcopy(v) for k, v in judge.items() if k != "ext"}
    judge_ext = {k: copy.deepcopy(v) for k, v in _obj(judge.get("ext")).items() if k != NS}
    if judge_ext:
        kept_judge["ext"] = judge_ext
    if kept_judge and kept_judge not in ({"kind": "model"}, {"kind": "user"}):
        rest["judge"] = kept_judge
    if preference.get("slate") != default_slate(inputs.get(a, "")):
        rest["slate"] = preference.get("slate")
    return _common(row, e, ("replaces", "assignment", "cost_usd"), rest)


def _signals(run_id: str, signals: Iterable[Any], where: str, batch: Batch, rubric_ref: str | None,
             components: set[str] | None, *, scope: bool = False) -> None:
    """Add the readings of one run's signals to `batch`. Verdicts and events are not readings and are skipped.

    `scope` is for run documents, whose signal ids are unique per document only (see `reading_row`)."""
    plain: dict[tuple[str, Any], list[tuple[float, int, dict]]] = {}

    def skip(signal: dict, reason: str) -> None:
        batch.not_used.append({"source": where, "kind": "signal", "id": signal.get("id"), "reason": reason})

    for n, signal in enumerate(signals):
        if not isinstance(signal, dict) or signal.get("kind") != "score":
            continue
        source = _obj(signal.get("source"))
        e = _obj(signal.get("ext")).get(NS)
        if source.get("kind") not in JUDGE_SOURCES or not isinstance(source.get("ref"), str) or not source["ref"]:
            skip(signal, "a score with no judge: source.kind is not judge or user, or source.ref is missing")
        elif signal.get("value") is None and _obj(e).get("level") is None:
            skip(signal, "a declared score with no value yet")
        elif isinstance(e, dict):
            row = reading_row(run_id, signal, rubric_ref, scope=scope)
            batch.scores.append(row)
            batch.source[("scores", str(row["id"]))] = where
        elif rubric_ref is None:
            skip(signal, "the signal names no rubric; give --rubric")
        elif components is not None and signal.get("name") not in components:
            skip(signal, f'"{signal.get("name")}" is not a component of the rubric')
        else:
            plain.setdefault((source["ref"], signal.get("name")), []).append(
                (_instant(signal.get("observed_at")), n, signal))
    for group in plain.values():
        earlier = None
        for _, _, signal in sorted(group, key=lambda entry: entry[:2]):
            row = reading_row(run_id, signal, rubric_ref, scope=scope)
            if earlier is not None:
                row["replaces"] = earlier
            earlier = row["id"]
            batch.scores.append(row)
            batch.source[("scores", str(row["id"]))] = where


def read_documents(documents: Iterable[tuple[str, Any]], *, rubric: dict | None = None,
                   pair_component: str | None = None, inputs: Mapping[str, str] | None = None) -> Batch:
    """Runs, readings and pairs of OCP run documents, given as `(file name, parsed document)`.

    A document that does not conform to OCP is a refusal that names the file and the place. `inputs` maps
    the runs already in the study to their inputs, so a preference can name a run imported earlier."""
    batch = Batch()
    rubric_ref = None if rubric is None else C.rubric_ref(rubric)
    components = None if rubric is None else {comp["id"] for comp in rubric["components"]}
    known = dict(inputs or {})
    docs = []
    for file, doc in documents:
        if not isinstance(doc, dict) or doc.get("kind") == "flow" or not isinstance(doc.get("run"), dict):
            raise C.ContractError(file, "", "not an OCP run document", "give run documents, one per file", "no_run")
        for finding in validate_doc(doc):
            if finding.level == "error":
                raise C.ContractError(file, finding.path, f"{finding.code}: {finding.message}",
                                      "fix the document (loopmath ocp validate names every problem)", "bad_ocp")
        row = run_row(doc, file)
        batch.runs.append(row)
        batch.source[("runs", row["run"])] = file
        known[row["run"]] = row["input"]
        docs.append((file, doc))
    seen: dict[str, dict] = {}
    for file, doc in docs:
        run = doc["run"]
        _signals(run["id"], run.get("signals") or [], file, batch, rubric_ref, components, scope=True)
        slate = _obj(run.get("slate"))
        fallback_time = run.get("ended_at") or run.get("started_at") or _obj(doc.get("producer")).get("emitted_at")
        for preference in run.get("preferences") or []:
            if not isinstance(preference, dict):
                continue
            key = repr((preference.get("slate"), preference.get("id")))  # a preference id is unique in its slate
            if seen.get(key) == preference:
                continue  # copied into every member's file: read once
            seen[key] = preference
            reason, row = None, None
            has_eval = isinstance(_obj(preference.get("ext")).get(NS), dict)
            members = preference.get("members")
            if not isinstance(members, list) and preference.get("slate") == slate.get("id"):
                members = slate.get("members")
            if not isinstance(members, list) or len(members) != 2:
                reason = "the preference does not name exactly two runs"
            elif preference.get("winner") not in (*members, "tie"):
                reason = "the winner is neither of the two runs nor tie"
            elif not has_eval and (members[0] not in known or members[1] not in known):
                reason = "the preference does not resolve to two runs of the study"
            elif not has_eval and known[members[0]] != known[members[1]]:
                reason = "the two runs have different inputs"
            elif not has_eval and rubric_ref is None:
                reason = "the preference names no rubric; give --rubric"
            else:
                row = pair_row(preference, members, known, rubric_ref, pair_component, fallback_time, scope=True)
                if row["observed_at"] is None:
                    reason = "the preference has no observed_at and its run has no time"
            if reason is not None or row is None:
                batch.not_used.append({"source": file, "kind": "preference", "id": preference.get("id"),
                                       "reason": reason})
                continue
            batch.pairs.append(row)
            batch.source[("pairs", str(row["id"]))] = file
    return batch


# ---------------------------------------------------------------------------------------------------
# harness-runs rows to rows

def read_rows(path: Any, *, rubric: dict | None = None) -> Batch:
    """Runs and readings of a harness-runs file. A row that breaks its format is a refusal by line.

    The run's `input` is `task_id`, its `arm` is `flow.arm` or `arm`, its steps are the stages (the last
    stage of each piece), else one step `final` with the row's model. Each `outcome.judges` item is a
    reading: `ext["dev.loopmath.eval"]` on the item states `component` and the members a signal holds. An item
    without it is read on the rubric's overall component, or its only component. The row's own
    `outcome.score` is a reading when the row names its `judge` and its `name` is a component."""
    file = Path(path).name
    try:
        rows = read_numbered(path)
    except ValueError as exc:
        where, problem = "", str(exc)
        if problem.startswith(f"{path}:"):  # "<path>:<line>: <problem>"
            line, _, rest = problem[len(f"{path}:"):].partition(": ")
            where, problem = (f"line {line}", rest) if line.isdigit() and rest else ("", problem)
        raise C.ContractError(file, where, problem, "fix the row, or check the path", "bad_rows") from None
    batch = Batch()
    rubric_ref = None if rubric is None else C.rubric_ref(rubric)
    components = None if rubric is None else {comp["id"] for comp in rubric["components"]}
    default = None
    if rubric is not None:
        default = C.overall_component(rubric) or (rubric["components"][0]["id"] if len(rubric["components"]) == 1
                                                  else None)
    for line, row in rows:
        where = f"{file} line {line}"
        run_id = run_id_for(row)
        flow, outcome = _obj(row.get("flow")), _obj(row.get("outcome"))
        arm = flow.get("arm") or row.get("arm")
        if not isinstance(arm, str) or not arm:
            raise C.ContractError(file, f"line {line}", "the row names no arm", "state arm, or flow.arm", "no_arm")
        steps: dict[str, dict] = {}
        costs = [_obj(row.get("cost")).get("usd")]
        for stage in row.get("stages") or []:
            piece = _ident(stage_piece(stage)[0])
            steps.pop(piece, None)
            steps[piece] = _step(piece, stage.get("model") or row.get("model") or "unknown",
                                 stage.get("effort") or row.get("effort"), stage.get("ext"))
            costs.append(_obj(stage.get("cost")).get("usd"))
        if not steps:
            steps[C.FINAL] = _step(C.FINAL, row.get("model") or "unknown", row.get("effort"), row.get("ext"))
        out: dict = {"run": run_id, "input": row["task_id"], "arm": arm}
        if isinstance(flow.get("split"), str) and flow["split"]:
            out["split"] = flow["split"]
        out["steps"] = list(steps.values())
        cost = _total(costs)
        if cost is not None:
            out["cost_usd"] = cost
        out["simulated"] = isinstance(_obj(row.get("ext")).get(SIMULATE_KEY), dict)
        batch.runs.append(out)
        batch.source[("runs", run_id)] = where
        at = outcome.get("observed_at") or row.get("ended_at") or row.get("started_at")
        signals = []
        if "score" in outcome and isinstance(outcome.get("judge"), str):
            signals.append({"id": f"{run_id}:score", "kind": "score", "name": outcome.get("name") or "judge",
                            "value": outcome["score"], "observed_at": at,
                            "source": {"kind": "judge", "ref": outcome["judge"]}})
        judges = outcome.get("judges") if isinstance(outcome.get("judges"), list) else []
        for n, item in enumerate(judges):
            if not isinstance(item, dict):
                continue
            e = _obj(_obj(item.get("ext")).get(NS))
            name = e.get("component") or default
            signal = {"id": e.get("id") or f"{run_id}:judge:{n}", "kind": "score", "name": name,
                      "value": item.get("score"), "observed_at": item.get("observed_at") or at,
                      "source": {"kind": "judge", "ref": item.get("id")}}
            if e:
                signal["ext"] = {NS: {k: v for k, v in e.items() if k not in ("component", "id")}}
            reason = None
            if name is None:
                reason = ("the judge score names no component; give --rubric" if rubric is None else
                          "the judge score names no component, and the rubric has no overall component and "
                          "more than one component")
            elif signal["observed_at"] is None and (signal["value"] is not None or e.get("level") is not None):
                reason = "the judge score has no observed_at and its row has no time"
            if reason is not None:
                batch.not_used.append({"source": where, "kind": "signal", "id": signal["id"], "reason": reason})
                continue
            signals.append(signal)
        _signals(run_id, signals, where, batch, rubric_ref, components)
    return batch


# ---------------------------------------------------------------------------------------------------
# Rows to OCP signals and preferences

def _ocp_time(text: str) -> tuple[str, str | None]:
    """`(the time as OCP writes it, the text as given when the two differ)`."""
    if _OCP_TIME.match(text):
        return text, None
    parsed = _dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    return parsed.isoformat(), text


def _eval_ext(row: dict, names: Iterable[str], kept: dict) -> tuple[dict, str]:
    e: dict = {name: row[name] for name in names if row.get(name) is not None}
    if row.get("cost_usd") is not None:
        e["cost_usd"] = row["cost_usd"]
    observed, raw = _ocp_time(row["observed_at"])
    if raw is not None:
        e["observed_at"] = raw
    own = {k: v for k, v in _obj(row.get("ext")).items() if k != KEPT}
    if own:
        e["ext"] = copy.deepcopy(own)
    for name, value in _obj(kept.get("eval")).items():
        e.setdefault(name, copy.deepcopy(value))
    return e, observed


def _lay(record: dict, kept: dict, e: dict, skip: Iterable[str]) -> dict:
    for name, value in kept.items():
        if name not in skip and name not in ("ext", "eval"):
            record.setdefault(name, copy.deepcopy(value))
    record.setdefault("tier", "reported")
    record["ext"] = {**copy.deepcopy(_obj(kept.get("ext"))), NS: e}
    return record


def signal(row: dict, source_kind: str = "judge") -> dict:
    """The OCP score signal of one reading (a normalised row of `scores.jsonl`)."""
    kept = _obj(_obj(row.get("ext")).get(KEPT))
    e, observed = _eval_ext(row, READING_MEMBERS, kept)
    value = row["value"]
    if isinstance(value, str):
        e["level"], value = value, None
    given = kept.get("id")  # the id another tool gave the signal in its run document
    out = {"id": given if isinstance(given, str) and given else row["id"], "kind": "score", "name": row["component"],
           "value": value, "observed_at": observed,
           "source": {**copy.deepcopy(_obj(kept.get("source")) or {"kind": source_kind}), "ref": row["judge"]}}
    if out["id"] != row["id"]:
        e["id"] = row["id"]
    return _lay(out, kept, e, ("source", "id"))


def preference(row: dict, slate: str, judge_kind: str = "model") -> dict:
    """The OCP preference of one pair (a normalised row of `pairs.jsonl`), on the slate `slate`."""
    kept = _obj(_obj(row.get("ext")).get(KEPT))
    e, observed = _eval_ext(row, PAIR_MEMBERS, kept)
    judge = copy.deepcopy(_obj(kept.get("judge")) or {"kind": judge_kind})
    judge["ext"] = {**_obj(judge.get("ext")), NS: {"id": row["judge"]}}
    given = kept.get("id")  # the id another tool gave the preference in its slate
    out = {"id": given if isinstance(given, str) and given else row["id"], "slate": kept.get("slate") or slate,
           "winner": "tie" if row["verdict"] == "tie" else row[row["verdict"]], "members": [row["a"], row["b"]],
           "judge": judge, "observed_at": observed}
    if out["id"] != row["id"]:
        e["id"] = row["id"]
    return _lay(out, kept, e, ("judge", "slate", "id"))


def export(runs: list[dict], scores: list[dict], pairs: list[dict], *, only: Iterable[str] | None = None,
           persons: Iterable[str] = ()) -> tuple[list[dict], list[dict]]:
    """`(entries, not_used)`: per run, what `eval scores export` prints, and the rows it could not place.

    An entry is `{"run", "signals", "preferences"}`, with `slate` when the run has preferences. Every
    reading and pair is written, corrected ones too, so nothing is lost on the way back. A preference is
    copied into the entry of both of its runs. Readings and pairs by a judge id in `persons` are written
    as a user's (`source.kind` `user`, `judge.kind` `user`); the others as a judge's and a model's.
    A reading or pair whose run is not in `runs` is listed in `not_used`."""
    persons = set(persons)
    by_id = {row["run"]: row for row in runs}
    wanted = list(by_id) if only is None else list(dict.fromkeys(only))
    entries = {name: {"run": name, "signals": [], "preferences": []} for name in wanted if name in by_id}
    used: dict[str, list[str]] = {}
    not_used = []
    for row in scores:
        if row["run"] not in by_id:
            not_used.append({"kind": "reading", "id": row["id"], "reason": f'no run "{row["run"]}" in the study'})
        elif row["run"] in entries:
            entries[row["run"]]["signals"].append(signal(row, "user" if row["judge"] in persons else "judge"))
    slates: dict[str, dict] = {}
    for row in pairs:
        missing = [name for name in (row["a"], row["b"]) if name not in by_id]
        if missing:
            not_used.append({"kind": "pair", "id": row["id"], "reason": f'no run "{missing[0]}" in the study'})
            continue
        own = [_obj(_obj(_obj(by_id[name].get("ext")).get(KEPT)).get("slate")) for name in (row["a"], row["b"])]
        shared = own[0] if own[0] and own[0].get("id") == own[1].get("id") else None
        record = preference(row, shared["id"] if shared else default_slate(by_id[row["a"]]["input"]),
                            "user" if row["judge"] in persons else "model")
        slate = slates.setdefault(record["slate"], copy.deepcopy(shared) if shared and shared["id"] == record["slate"]
                                  else {"id": record["slate"], "members": []})
        for name in (row["a"], row["b"]):
            if name not in slate["members"]:
                slate["members"].append(name)
            if name in entries:
                entries[name]["preferences"].append(record)
                if record["slate"] not in used.setdefault(name, []):
                    used[name].append(record["slate"])
    out = []
    for name in wanted:
        if name not in entries:
            not_used.append({"kind": "run", "id": name, "reason": "no such run in the study"})
            continue
        entry = entries[name]
        names = used.get(name, [])
        if names:
            entry["slate"] = slates[names[0]]
            if len(names) > 1:
                entry["other_slates"] = [slates[other] for other in names[1:]]
        out.append(entry)
    return out, not_used


def merge(doc: dict, entry: dict) -> dict:
    """A copy of the OCP run document `doc` with one export entry merged in: the signals and preferences
    that are not there yet, by id, and the slate when the run has none."""
    out = copy.deepcopy(doc)
    run = out["run"]
    if entry.get("run") != run.get("id"):
        raise ValueError(f"the entry is for run {entry.get('run')!r}, the document is run {run.get('id')!r}")
    for name in ("signals", "preferences"):
        have = {record.get("id") for record in run.get(name) or [] if isinstance(record, dict)}
        new = [copy.deepcopy(record) for record in entry.get(name) or [] if record["id"] not in have]
        if new:
            run[name] = list(run.get(name) or []) + new
    if entry.get("slate") is not None and "slate" not in run:
        run["slate"] = copy.deepcopy(entry["slate"])
    return out
