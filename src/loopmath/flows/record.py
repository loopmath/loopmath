"""`loopmath flow record` and `loopmath flow score`: a flow's runs into the store, and one judge's score of a run.

Record reads every row or run document and checks it against the flow first: the flow's id, version and digest,
the arm and its configuration, the input, the simulated marker, the acceptance rule and the store's own rules.
One refusal writes nothing and names every refused item. Then, under the store's lock, each new run is stored
finished, and one background refit starts when a run was new. A run already recorded with the same content is
`unchanged`; one with other content is refused, never replaced. Record does not advance the plan.

Score writes one score signal of a flow judge into a stored run of the flow: pending for an open run, late for a
finished one (receipts re-scored, one background refit). A second score of the same run, judge and score name is
refused unless `--replace`, which adds the new score as the newest reading and deletes nothing.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from ..harness_runs.ocp import FLOW_EXT, flow_row_findings, known_configurations, to_ocp
from ..harness_runs.rows import names_flow, read_numbered
from ..ocp import conformance, finite
from ..ocp.emit import validate_strict
from ..store import run_names
from ..store.commands import _differs, _prepare_import, is_simulated
from ..store.home import Store, StoreError
from ..store.ids import new_id, now_iso, observed_iso
from ..store.runs import dollars_past_float
from ..types import Signal
from .folder import PLAN_FILE, FlowFolder
from .records import judge_scale, judge_score

UNPLANNED = "unplanned"
NO_FLOW = ("names no flow; flow record takes this flow's rows (convert others with loopmath adapt "
           "harness-runs)")
RUN_IMPORT = "store it with loopmath run import"
INT_DIGITS = 4300  # the interpreter's limit on reading an integer, never lifted
INTEGER = re.compile(r"[+-]?[0-9]+")
DECIMAL = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")
SCORE_KINDS = ("judge", "user")


class Refused(ValueError):
    """Items refused by `prepare`, each `{item, run, reason}`; nothing was written."""

    def __init__(self, refused: list[dict[str, Any]]) -> None:
        super().__init__("; ".join(f"{r['item']}: {r['reason']}" for r in refused))
        self.refused = refused


# ------------------------------------------------------------------ record: prepare

def _item(item: str, run: Any, reason: str) -> dict[str, Any]:
    return {"item": item, "run": run if isinstance(run, str) else None, "reason": reason}


def _json_equal(a: Any, b: Any) -> bool:
    """Equal as JSON values: `1` equals `1.0`, a boolean equals only a boolean, members and items compared alike."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _score_name(rule: Any) -> str:
    score = rule.get("score") if isinstance(rule, dict) else None
    name = score.get("name") if isinstance(score, dict) else None
    return repr(name) if isinstance(name, str) else "no score"


def _rule_problem(folder: FlowFolder, run: dict[str, Any]) -> str | None:
    """Why the run's acceptance rule is not the flow's, or None; an absent rule is the flow's (`_normalize`)."""
    if "acceptance_rule" not in run:
        return None
    given, own = run["acceptance_rule"], folder.flow.acceptance_rule
    if _json_equal(given, own):
        return None
    if isinstance(given, dict):
        members = sorted(k for k in given.keys() | own.keys() if k not in given or k not in own
                         or not _json_equal(given[k], own[k]))
    else:
        members = ["(the rule)"]
    return (f"run.acceptance_rule (score {_score_name(given)}) is not the flow's (score {_score_name(own)}); they "
            f"differ in {', '.join(members)}; a run of the flow is judged by the flow's rule")


def _arm(folder: FlowFolder, run: dict[str, Any], allow_unplanned: bool) -> tuple[str | None, str | None, str | None]:
    """(the arm to record, the arm it named when that is `unplanned`, why it is refused). A known arm (none is
    base) is recorded when the run's configuration is that arm's. `--allow-unplanned` takes an arm the plan lacks,
    `unplanned` itself, any arm but base with no plan, and a configuration that is no arm's; never a stale plan or
    another arm's configuration."""
    ref = run["flow"]
    given = named = ref.get("arm")
    if named is not None and not isinstance(named, str):
        return None, None, f"run.flow.arm {named!r} is not a string"
    configuration = run.get("configuration") if isinstance(run.get("configuration"), dict) else {}
    cid = configuration.get("id") if isinstance(configuration.get("id"), str) else None
    known = known_configurations(folder)
    plan = folder.plan
    if given not in (None, "base") and folder.plan_stale:
        try:
            folder.configuration(given)
        except ValueError as exc:
            return None, None, str(exc)
    if given in (None, "base") or (plan is not None and given in {a.id for a in plan.arms}):
        try:
            arm, declared = folder.configuration(given)
        except ValueError as exc:
            return None, None, str(exc)
        if cid == declared.get("id"):
            return arm, None, None
        if cid in known:
            return None, None, (f"run.flow.arm is {arm!r}, but run.configuration.id {cid!r} is arm {known[cid]!r}'s "
                                f"configuration; the arm and the configuration must agree")
        why = (f"run.configuration.id {cid!r} is no arm's configuration (arm {arm!r} has {declared.get('id')!r})")
        named_arm = arm if named is not None else None
    elif given == UNPLANNED:
        held = _held_named_arm(run)
        why = "run.flow.arm is 'unplanned'" + (f" (it named arm {held!r})" if held is not None else "")
        if held is not None and plan is None:
            why += f", and there is no {PLAN_FILE} in {folder.dir}; run loopmath flow plan, or"
        named_arm = None
    elif plan is None:
        why = f"run.flow.arm is {given!r}, but there is no {PLAN_FILE} in {folder.dir}; run loopmath flow plan, or"
        named_arm = given
    else:
        arms = ", ".join(["base"] + [a.id for a in plan.arms if a.id != "base"])
        why, named_arm = f"run.flow.arm {given!r} is not an arm of {folder.dir / PLAN_FILE} (arms: {arms})", given
    if not allow_unplanned:
        joint = " " if why.endswith(", or") else "; "
        return None, None, f"{why}{joint}--allow-unplanned records it under arm 'unplanned'"
    if cid in known:
        return None, None, (f"{why}, but run.configuration.id {cid!r} is arm {known[cid]!r}'s configuration; name "
                            f"arm {known[cid]!r}")
    return UNPLANNED, named_arm, None


def _held_named_arm(run: dict[str, Any]) -> Any:
    """The arm an `unplanned` run named, as `run.flow.ext["dev.loopmath.flow"].named_arm` holds it, or None."""
    ext = run["flow"].get("ext")
    held = ext.get(FLOW_EXT) if isinstance(ext, dict) else None
    return held.get("named_arm") if isinstance(held, dict) else None


def _flow_ext_problem(run: dict[str, Any], named_arm: str | None) -> str | None:
    """Why `run.flow.ext` cannot be stored as given, or cannot keep the arm the run named, or None: an ext that is
    no object; where the named arm is to be kept, a `dev.loopmath.flow` value that is no object or a `named_arm`
    there that differs (null too). Record never replaces what the producer wrote; a value it does not write to is
    kept as it is."""
    flow = run["flow"]
    if "ext" in flow and not isinstance(flow["ext"], dict):
        return f"run.flow.ext {flow['ext']!r} is not an object"
    if named_arm is None or FLOW_EXT not in flow.get("ext", {}):
        return None
    held = flow["ext"][FLOW_EXT]
    if not isinstance(held, dict):
        return (f"run.flow.ext[{FLOW_EXT!r}] {held!r} is not an object, so the arm the run named, {named_arm!r}, "
                "cannot be kept there as named_arm")
    if "named_arm" in held and held["named_arm"] != named_arm:
        return f"run.flow.ext[{FLOW_EXT!r}].named_arm is {held['named_arm']!r}, but the run names arm {named_arm!r}"
    return None


def _normalize(folder: FlowFolder, doc: dict[str, Any], arm: str, named_arm: str | None) -> None:
    """The stored copy: the flow's acceptance rule when the run gives none, and arm `unplanned` with the arm it
    named kept in `run.flow.ext["dev.loopmath.flow"]` (other members and namespaces kept)."""
    run = doc["run"]
    if "acceptance_rule" not in run:
        run["acceptance_rule"] = copy.deepcopy(folder.flow.acceptance_rule)
    if arm == UNPLANNED:
        run["flow"]["arm"] = UNPLANNED
        if named_arm is not None:  # `_flow_ext_problem` lets through only an object, or none, at each level
            ext = run["flow"].setdefault("ext", {})
            ext[FLOW_EXT] = {**ext.get(FLOW_EXT, {}), "named_arm": named_arm}


def _check(folder: FlowFolder, store: Store, item: str, doc: Any, *, allow_unplanned: bool,
           allow_simulated: bool) -> tuple[dict[str, Any] | None, str | None]:
    """One run document checked against the flow and the store, writing nothing: (what to store, None) or (None,
    why it is refused). The flow, digest, arm and `run.flow.ext`, input, simulated marker, rule and run id in that
    order, each in its own words, then the store's import checks, the strict and cross-file validation of the copy
    to store, a run that states no end, dollars no float holds, and a run of that id already stored."""
    if not isinstance(doc, dict) or not isinstance(doc.get("run"), dict):
        return None, "not an OCP run document (no run object)"
    run = doc["run"]
    if str(doc.get("ocp", "")) != "0.4" or not isinstance(run.get("flow"), dict):
        return None, (f"an OCP {doc.get('ocp')!r} run with no run.flow object is no run of a flow; {RUN_IMPORT}"
                      if not isinstance(run.get("flow"), dict) else
                      f"declares OCP {doc.get('ocp')!r}; a run of a flow is OCP '0.4'")
    ref = run["flow"]
    if ((ref.get("id"), ref.get("version")) != (folder.flow.id, folder.flow.version)
            or isinstance(ref.get("version"), bool)):
        return None, (f"run.flow names flow {ref.get('id')!r} version {ref.get('version')!r}; the flow given is flow "
                      f"{folder.flow.id!r} version {folder.flow.version} ({folder.path})")
    if "digest" not in ref:
        return None, f"run.flow gives no digest; the flow's is {folder.digest}"
    if ref["digest"] != folder.digest:
        return None, (f"run.flow.digest {ref['digest']!r} is not the flow's digest {folder.digest}: the run was made "
                      "for another revision of the flow")
    arm, named_arm, why = _arm(folder, run, allow_unplanned)
    if why is None:
        why = _flow_ext_problem(run, named_arm)
    if why is not None:
        return None, why
    task = run.get("task") if isinstance(run.get("task"), dict) else {}
    ids = folder.flow.inputs.ids
    if ids is not None and task.get("id") not in ids:
        return None, (f"task.id {task.get('id')!r} is not among the flow's inputs.ids ({len(ids)} ids)")
    if is_simulated(doc) and not allow_simulated:
        return None, (f"run {run.get('id')} is simulated (run.ext[{conformance.SIMULATE_KEY!r}]); "
                      "--allow-simulated records simulated runs")
    why = _rule_problem(folder, run)
    if why is not None:
        return None, why
    if run_names.file_stem(run.get("id")) is None:
        return None, f"run.id {run.get('id')!r} cannot be stored: a run id {run_names.RULE}"
    try:
        prepared = _prepare_import(store, doc, item)
    except (ValueError, StoreError) as exc:
        return None, str(exc)
    _normalize(folder, prepared["doc"], arm, named_arm)
    errors = [f for f in validate_strict(prepared["doc"]) if f["level"] == "error"]
    if not errors:
        errors = [f._asdict() for f in conformance.validate_many([folder.doc, prepared["doc"]])[1]
                  if f.level == "error"]
    if errors:
        return None, "; ".join(f"{e['code']} at {e['path']}: {e['message']}" for e in errors[:3]) + (
            f" and {len(errors) - 3} more" if len(errors) > 3 else "")
    if _end(prepared["doc"]) is None:
        return None, (f"run {run.get('id')} states no end (run.ended_at, or an attempt's ended_at); flow record "
                      "stores a run finished at its own end")
    where = dollars_past_float(prepared["doc"])
    if where is not None:  # it is stored finished, and the store refuses that on its commit
        return None, f"{where} is past what a float holds, so the run cannot be recorded finished"
    if prepared["existed"] and prepared["differs"]:
        return None, (f"run {prepared['run']} is recorded with other content ({', '.join(prepared['differs'][:6])}); "
                      "flow record does not replace a recorded run")
    prepared.update(arm=arm, input=task.get("id"), ids_listed=ids is not None)
    return prepared, None


def _rows(folder: FlowFolder, path: Path) -> tuple[list[tuple[str, Any]], list[dict[str, Any]], list[str]]:
    """(item and converted document per row, refused rows, the converter's findings): rows read as `adapt
    harness-runs` reads them, each converted with the flow, an arm the plan lacks as `unplanned` (the check names
    `--allow-unplanned`). A row whose document has no `run.ended_at` (neither the row nor its last stage states an
    end) is refused."""
    from ..harness_runs.commands import _mtime

    numbered = read_numbered(path)
    default = _mtime(path)
    docs, refused, findings = [], [], []
    for n, row in numbered:
        item = f"{path}:{n}"
        if not names_flow(row):
            refused.append(_item(item, row.get("id"), NO_FLOW))
            continue
        try:
            doc = to_ocp(row, base_dir=path.parent, observed_default=default, flow=folder, unplanned=True)
        except (OSError, ValueError) as exc:
            refused.append(_item(item, row.get("id"), str(exc)))
            continue
        if not doc["run"].get("ended_at"):
            refused.append(_item(item, row.get("id"), f"row {row.get('id')} states no end (ended_at, or its last "
                                 "stage's ended_at); flow record stores a run finished at its own end"))
            continue
        docs.append((item, doc))
        findings += [f"{item}: {line}" for line in flow_row_findings(row, folder)]
    return docs, refused, findings


def _documents(files: list[str]) -> tuple[list[tuple[str, Any]], list[dict[str, Any]]]:
    docs, refused = [], []
    for name in files:
        try:
            docs.append((name, finite.loads(Path(name).expanduser().read_text(encoding="utf-8"))))
        except FileNotFoundError:
            refused.append(_item(name, None, "no such file"))
        except finite.NotFinite as exc:
            refused.append(_item(name, None, str(exc)))
        except (OSError, ValueError) as exc:
            refused.append(_item(name, None, f"cannot be read as JSON: {exc}"))
    return docs, refused


def prepare(folder: FlowFolder, store: Store, *, rows: str | None = None, files: list[str] | None = None,
            allow_unplanned: bool = False, allow_simulated: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    """Every row or document read, converted and checked, nothing written: (the runs to store, each with `item`,
    the converter's findings). Raises Refused naming every refused item, and ValueError for a rows file that cannot
    be read."""
    if rows is not None:
        docs, refused, findings = _rows(folder, Path(rows).expanduser())
    else:
        (docs, refused), findings = _documents(files or []), []
    ready, first = [], {}
    for item, doc in docs:
        prepared, why = _check(folder, store, item, doc, allow_unplanned=allow_unplanned,
                               allow_simulated=allow_simulated)
        if why is not None:
            run = doc.get("run") if isinstance(doc, dict) else None
            refused.append(_item(item, run.get("id") if isinstance(run, dict) else None, why))
            continue
        if prepared["run"] in first:
            refused.append(_item(item, prepared["run"], f"run {prepared['run']} is also given by "
                                 f"{first[prepared['run']]}; give each run once"))
            continue
        first[prepared["run"]] = item
        ready.append({**prepared, "item": item})
    if refused:
        raise Refused(refused)
    return ready, findings


# ------------------------------------------------------------------ record: commit

def _end(doc: dict[str, Any]) -> str | None:
    """Where a stored run's `ended_at` comes from: the run's own, else its attempts' latest; None when neither
    states one (`_check` refuses that run)."""
    if doc["run"].get("ended_at"):
        return "run"
    if any(isinstance(a, dict) and a.get("ended_at") for a in doc.get("attempts") or []):
        return "attempts"
    return None


def commit(store: Store, ready: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Store each prepared run finished, under the store's lock, its collisions checked again first: one per run,
    `{run, arm, input, import, ended_at_from}`, `import` new or unchanged. Not a transaction: an I/O failure
    partway raises OSError or StoreError naming the run it stopped at and those stored before it."""
    from ..store.home import StoreError

    out = []
    with store.lock():
        for p in ready:
            if store.exists(p["run"]):
                differs = _differs(store.run_doc(p["run"]), p["record"], p["doc"])
                if differs:
                    raise Refused([_item(p["item"], p["run"], f"run {p['run']} was recorded with other content while "
                                         f"this ran ({', '.join(differs[:6])}); nothing was written")])
                p["import"] = "unchanged"
            else:
                p["import"] = "new"
        stored: list[str] = []
        for p in ready:
            end = _end(p["doc"]) if p["import"] == "new" else None
            if p["import"] == "new":
                try:
                    store.import_run(p["doc"], finished=True)
                except (OSError, StoreError) as exc:
                    kind = OSError if isinstance(exc, OSError) else StoreError  # some take no message alone
                    raise kind(f"stopped at run {p['run']} ({exc}); stored before it: "
                               f"{', '.join(stored) or 'none'}") from exc
                stored.append(p["run"])
            out.append({"run": p["run"], "arm": p["arm"], "input": p["input"], "import": p["import"],
                        "ended_at_from": end})
    return out


def refit(store: Store, recorded: list[dict[str, Any]]) -> dict[str, Any]:
    """One background refit when a run was new; none when every run was unchanged."""
    if not any(r["import"] == "new" for r in recorded):
        return {"started": False, "reason": "no run was new"}
    from ..store.fitjob import spawn_fit

    return spawn_fit(store.home)


# ------------------------------------------------------------------ score

def parse_value(text: str) -> int | float:
    """VALUE as a number: an integer literal (sign, ASCII digits) exactly, a decimal (point or exponent) as a
    finite float. Raises ValueError naming the problem: not a number, not finite, more digits than an integer may
    have."""
    text = text.strip()
    if INTEGER.fullmatch(text):
        digits = len(text.lstrip("+-"))
        if digits > INT_DIGITS:
            raise ValueError(f"VALUE has {digits} digits; an integer of more than {INT_DIGITS} digits cannot be read")
        return int(text)
    if not DECIMAL.fullmatch(text):
        raise ValueError(f"VALUE {text!r} is not a number")
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"VALUE {text!r} is not a finite number")
    return value


def parse_observed(text: str) -> str:
    """`--observed-at` in its stored form, UTC with a `Z` (`observed_iso`): an ISO 8601 time that states its offset
    (`Z` or `+hh:mm`) and that a UTC time holds."""
    return observed_iso(text, flag="--observed-at", need_offset=True)


def _under_key(doc: dict[str, Any], judge: str, name: str) -> list[dict[str, Any]]:
    """The run's score signals of `judge` under `name`, from a judge or a person, whatever their value."""
    from ..belief.outcome import signal_list

    run = doc.get("run") if isinstance(doc.get("run"), dict) else {}
    return [s for s in signal_list(run) if isinstance(s, dict) and s.get("kind") == "score" and s.get("name") == name
            and isinstance(s.get("source"), dict) and s["source"].get("kind") in SCORE_KINDS
            and s["source"].get("ref") == judge]


def score(folder: FlowFolder, store: Store, run: str, judge_id: str, value_text: str, *, replace: bool = False,
          observed_at: str | None = None, at_attempt: str | None = None) -> dict[str, Any]:
    """Write one score of flow judge `judge_id` into stored run `run`: `{run, judge, signal, value, replaced,
    state}`, `state` pending or late, `replaced` the ids of the scores under the key before it (with --replace), or
    None. Raises ValueError naming why, with nothing written."""
    from ..belief.outcome import parse_ts
    from ..store.home import NotFound

    judges = {j.id: j for j in folder.flow.judges or ()}
    judge = judges.get(judge_id)
    if judge is None:
        raise ValueError(f"--judge {judge_id!r} is no judge of flow {folder.flow.id!r} (judges: "
                         f"{', '.join(judges) or 'none'})")
    value = parse_value(value_text)
    scale = judge_scale(judge)
    if scale == "fraction" and not 0 <= value <= 1:
        raise ValueError(f"VALUE {value_text.strip()} is outside [0, 1], the scale of judge {judge_id!r} (fraction)")
    if scale == "linear" and judge.range is not None and not judge.range[0] <= value <= judge.range[1]:
        lo, hi = judge.range
        raise ValueError(f"VALUE {value_text.strip()} is outside [{lo}, {hi}], the range of judge {judge_id!r}")
    name = judge_score(judge, folder.flow.objective)
    if not isinstance(name, str):
        raise ValueError(f"judge {judge_id!r} names no score and the flow's objective names none")
    stamp = parse_observed(observed_at) if observed_at is not None else None
    if not store.exists(run):  # before the lock, which makes the home
        raise ValueError(f"no run {run} in the store at {store.home}")
    with store.lock():  # read, check and write under one lock; add_signal takes it again
        try:
            doc = store.run_doc(run)
        except NotFound:
            raise ValueError(f"no run {run} in the store at {store.home}") from None
        ref = (doc.get("run") or {}).get("flow")
        ref = ref if isinstance(ref, dict) else {}
        if (ref.get("id"), ref.get("version"), ref.get("digest")) != (folder.flow.id, folder.flow.version,
                                                                       folder.digest):
            raise ValueError(f"run {run} is no run of flow {folder.flow.id!r} version {folder.flow.version} (digest "
                             f"{folder.digest}): its run.flow names flow {ref.get('id')!r} version "
                             f"{ref.get('version')!r} (digest {ref.get('digest')!r})")
        attempts = [a.get("id") for a in doc.get("attempts") or [] if isinstance(a, dict)]
        if at_attempt is not None and at_attempt not in attempts:
            raise ValueError(f"--at-attempt {at_attempt!r} is no attempt of run {run} (attempts: "
                             f"{', '.join(map(str, attempts)) or 'none'})")
        prior = _under_key(doc, judge_id, name)
        if prior and not replace:
            raise ValueError(f"run {run} already has a score of judge {judge_id!r} under {name!r} "
                             f"({', '.join(str(s.get('id')) for s in prior)}); --replace adds this one as the newest "
                             "reading")
        if replace and not prior:
            raise ValueError(f"--replace: run {run} has no score of judge {judge_id!r} under {name!r} to replace")
        stamp = stamp if stamp is not None else now_iso()
        instant = parse_ts(stamp)  # the stored form's instant, so a fraction utc_iso drops is not later
        times = [(t, s.get("observed_at")) for s in prior if (t := parse_ts(s.get("observed_at"))) is not None]
        newest = max(times, key=lambda p: p[0]) if times else None
        if replace and newest is not None and not instant > newest[0]:
            raise ValueError(f"--replace: the new score's time {stamp} is not later than the newest score's under "
                             f"{name!r} ({newest[1]}), so it would not be the reading; give a later --observed-at")
        extra = {"ext": {"dev.loopmath.judge": {"range": list(judge.range)}}} \
            if scale == "linear" and judge.range is not None else {}
        signal = Signal(id=new_id("sig"), run=run, kind="score", name=name, value=value, better="higher", scale=scale,
                        at_attempt=at_attempt if at_attempt is not None else (attempts[-1] if attempts else None),
                        observed_at=stamp, source={"kind": "user" if judge.kind == "person" else "judge",
                                                   "ref": judge_id}, tier="reported", extra=extra)
        state = store.add_signal(run, signal)
    return {"run": run, "judge": judge_id, "signal": signal.id, "value": value,
            "replaced": [s.get("id") for s in prior] if replace else None, "state": state}
