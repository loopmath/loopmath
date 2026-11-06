"""`loopmath flow status FLOW`: what the store's runs of the flow show, read with one judges fit on the plan brought
up to date and not saved. It writes nothing and exits 0, with `plan_problem` when the plan cannot be read; `--json`
is `loopmath.flow-status/1` with `STATUS_KEYS`. Runs are simulated from the example's truth, seeds stated. What
the store refuses to hold (a score that is not a number, `run.signals` not a list, an attempt that is not an object
or of dollars past a float) goes to `status.read` directly."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from loopmath import cli, compare
from loopmath.belief.outcome import FAIL, PASS, SIGNALS_NOT_LIST, parse_ts, unreadable_signals
from loopmath.flows import check as check_mod
from loopmath.flows import judges, load, simulate, status
from loopmath.flows import plan as plan_mod
from loopmath.flows.commands import _usable_plan
from loopmath.flows.next import next_runs
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.ocp.emit import validate_strict
from loopmath.output import counted, fmt_usd
from loopmath.store import Store, fitjob
from loopmath.workflows import ocp as workflows_ocp
from loopmath.workflows.format import catalog
from tests.flows.test_flow_judges import calibrated_runs
from tests.flows.test_flow_record import _own

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
ARM = "merge-extract-synthesize"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone(timedelta(hours=-8)))


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


@pytest.fixture
def lb(tmp_path, capsys) -> Path:
    """The example flow with the plan `flow plan` makes."""
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder)
    assert _flow(capsys, "plan", str(folder), "--home", str(tmp_path / "home"))[0] == 0
    return folder


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "home"


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _status(capsys, folder: Path, home: Path, *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "status", str(folder), "--home", str(home), *more)


def _strict(text: str):
    def refuse(name):
        raise AssertionError(f"not standard JSON: {name}")

    return json.loads(text, parse_constant=refuse)


def _plain(value):
    """`value` as its JSON reads back."""
    return json.loads(json.dumps(value))


def _files(folder: Path, home: Path) -> dict[str, bytes]:
    return {p.relative_to(p.anchor).as_posix(): p.read_bytes() for root in (folder, home) if root.exists()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "lock"}


def _screen(folder: Path, n: int, seed: int = 5) -> list[dict]:
    """The screen's first `n` runs (`next_runs` with seed 1), simulated with `seed`."""
    flow = load(folder)
    return simulate.simulate(flow, arm="base", n=0, seed=seed, truth=TRUTH, specs=next_runs(flow, n=n, seed=1))


def _stored(home: Path, docs: list[dict]) -> None:
    store = Store(home)
    for doc in docs:
        store.import_run(doc)


def _store_docs(home: Path, flow_id: str) -> list[dict]:
    """The store's runs of the flow id, as status reads them."""
    store = Store(home)
    return [store.run_doc(row["run"]) for row in store.flow_runs(flow_id)]


def _arm(doc: dict) -> str:
    return doc["run"]["flow"].get("arm") or "base"


def _arm_order(folder: Path, docs: list[dict]) -> list[str]:
    """The plan's arms in its order, `base` first, then any other sorted: those the runs hold."""
    planned = ["base"] + [a["id"] for a in json.loads((folder / "plan.json").read_text(encoding="utf-8"))["arms"]
                          if a["id"] != "base"]
    present = {_arm(d) for d in docs}
    return [a for a in planned if a in present] + sorted(present - set(planned))


def _fit(folder: Path, docs: list[dict]):
    flow = load(folder)
    return plan_mod.quality_and_fit(flow.flow, plan_mod.flow_runs(flow, docs), None, None)


def _view(fit) -> dict | None:
    """The fit as status shows it: `FIT_KEYS`, `refused` and `anchor`, each judge `JUDGE_KEYS`."""
    if not fit:
        return None
    view = {k: fit.get(k) for k in judges.FIT_KEYS + ("refused", "anchor")}
    view["judges"] = [{k: j.get(k) for k in judges.JUDGE_KEYS} for j in fit["judges"]]
    return _plain(view)


# Keys and a fresh plan --------------------------------------------------------------------------------------------

def test_a_fresh_plan_with_no_runs_shows_every_experiment_planned_and_nothing_else(capsys, lb, home):
    plan = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    before = _files(lb, home)
    code, out, err = _status(capsys, lb, home, "--json")
    report = _strict(out)
    assert code == 0 and err == "" and list(report) == list(status.STATUS_KEYS)
    assert report["flow"] == {"id": "lit-brief", "version": 1, "digest": load(lb).digest}
    assert (report["plan"], report["plan_problem"], report["runs"], report["other_revisions"]) == (plan["id"], None,
                                                                                                  0, 0)
    assert (report["cost"], report["checks"], report["quality"], report["judges"]) == ([], [], [], None)
    assert (report["wait"], report["adopted"], report["notes"]) == (None, None, [])
    assert report["dollars"] == {"runs": 0, "priced_runs": 0, "attempts": 0, "priced": 0, "usd_known": 0.0,
                                 "usd": 0.0, "reason": None}
    assert [e["id"] for e in report["experiments"]] == [e["id"] for e in plan["experiments"]]
    assert all(list(e) == list(check_mod.STATE_KEYS) and (e["state"], e["pairs"], e["verdict"]) == ("planned", 0, None)
               for e in report["experiments"])
    code, out, err = _status(capsys, lb, home)
    assert code == 0 and err == ""
    assert out.splitlines() == [
        f"flow lit-brief version 1 (digest {load(lb).digest})", f"plan {plan['id']}", "runs: 0",
        "cost per step and model:", "checks:", "quality per arm (pooled):", "judges:", "  no run holds a judge's score",
        "experiments:",
        *(f"  {e['id']} ({e['kind']}{', arms ' + ', '.join(e['arms']) if e['arms'] else ''}): planned, 0 pairs, "
          f"{e['pairs_left']} left" for e in report["experiments"]),
        f"dollars: 0 runs, 0 priced; {fmt_usd(0.0)}"]
    assert _files(lb, home) == before


# A recorded screen ------------------------------------------------------------------------------------------------

def test_a_recorded_screen_reads_as_its_runs_show_and_status_writes_nothing(capsys, lb, home, tmp_path):
    """The screen's 32 runs (seed 5) recorded with `flow record`: each section against the runs read on their own."""
    paths = []
    for k, doc in enumerate(_screen(lb, 32)):
        paths.append(tmp_path / "docs" / f"run-{k:02d}.json")
        paths[-1].parent.mkdir(exist_ok=True)
        paths[-1].write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = _flow(capsys, "record", str(lb), "--ocp", *map(str, paths), "--allow-simulated", "--home",
                           str(home))
    assert code == 0, err
    flow, docs = load(lb), _store_docs(home, "lit-brief")
    quality, fit = _fit(lb, docs)
    before = _files(lb, home)
    code, out, err = _status(capsys, lb, home, "--json")
    report = _strict(out)
    assert code == 0 and err == "" and report["runs"] == len(docs) == 32 and report["other_revisions"] == 0
    arms = _arm_order(lb, docs)
    by_arm = {arm: [d for d in docs if _arm(d) == arm] for arm in arms}
    # quality: per arm, the mean of the finite pooled q
    assert [r["arm"] for r in report["quality"]] == arms
    for row in report["quality"]:
        qs = [q for d in by_arm[row["arm"]] for q in [judges.combined_quality(flow.flow, d, fit)[0]]
              if q is not None and math.isfinite(q)]
        assert list(row) == list(status.QUALITY_KEYS) and row["runs"] == len(by_arm[row["arm"]])
        assert (row["scored"], row["pooled_mean"]) == (len(qs), math.fsum(qs) / len(qs) if qs else None)
    assert report["judges"] == _view(fit)
    # checks: each run once, by its verdict of the check's name; the inline workflow's gate after `critic` is a
    # check of that name in each arm whose runs have it, beside the rule's `claims-check`
    gated = {arm for arm in arms if any("critic" in d["run"]["configuration"]["workflow"]["control"]["gates"]
                                        for d in by_arm[arm])}
    assert gated and {(r["check"], r["arm"]) for r in report["checks"]} == (
        {("claims-check", arm) for arm in arms} | {("critic", arm) for arm in gated})
    for row in report["checks"]:
        newest = [max(v, key=lambda s: parse_ts(s["observed_at"])) if v else None
                  for d in by_arm[row["arm"]] for v in [[s for s in d["run"]["signals"] if s.get("kind") == "verdict"
                                                         and s.get("name") == row["check"]]]]
        values = [s["value"].lower() if s is not None else None for s in newest]
        decided = sum(v in PASS for v in values) + sum(v in FAIL for v in values)
        assert list(row) == list(status.CHECK_ROW_KEYS) and row["runs"] == len(by_arm[row["arm"]])
        assert (row["pass"], row["fail"], row["missing"], row["other"]) == (
            sum(v in PASS for v in values), sum(v in FAIL for v in values), values.count(None), 0)
        assert row["rate"] == (row["pass"] / decided if decided else None)
    # cost: one row per step and model, summing to the dollars
    attempts = [a for d in docs for a in d["attempts"]]
    pieces = [p["id"] for p in flow.flow.configuration["workflow"]["pieces"]]
    order = {p: n for n, p in enumerate(pieces)}
    assert [(r["step"], r["model"]) for r in report["cost"]] == sorted(
        Counter((a["node"], a["model"]["id"]) for a in attempts),
        key=lambda k: (order.get(k[0], len(order)), "" if k[0] in order else k[0], k[1]))
    for row in report["cost"]:
        own = [a["cost"]["usd"] for a in attempts if (a["node"], a["model"]["id"]) == (row["step"], row["model"])]
        assert list(row) == list(status.COST_KEYS)
        assert row == {"step": row["step"], "model": row["model"], "attempts": len(own), "priced": len(own),
                       "usd_known": math.fsum(own), "usd": math.fsum(own), "reason": None}
    total = math.fsum(a["cost"]["usd"] for a in attempts)
    assert report["dollars"] == {"runs": 32, "priced_runs": 32, "attempts": len(attempts), "priced": len(attempts),
                                 "usd_known": total, "usd": total, "reason": None}
    assert math.fsum(r["usd_known"] for r in report["cost"]) == pytest.approx(total, rel=1e-12)
    # the plan as advance brings it up to date, unsaved
    advanced = plan_mod.advance(flow, docs, quality=quality, fit=fit)
    assert report["plan"] == advanced.id
    assert report["experiments"] == _plain(check_mod.experiment_states(dataclasses.replace(flow, plan=advanced), docs,
                                                                      quality=quality, fit=fit))
    assert report["notes"] == []
    code, out, err = _status(capsys, lb, home)
    assert code == 0 and err == "" and out.splitlines()[:3] == [
        f"flow lit-brief version 1 (digest {flow.digest})", f"plan {advanced.id}", "runs: 32"]
    assert _files(lb, home) == before


def test_one_judges_fit_serves_every_section(capsys, lb, home, monkeypatch):
    _stored(home, _screen(lb, 4))
    calls, real = [], judges.fit
    monkeypatch.setattr(judges, "fit", lambda flow, docs, *a, **kw: calls.append(len(docs)) or real(flow, docs, *a,
                                                                                                     **kw))
    assert _status(capsys, lb, home, "--json")[0] == 0
    assert calls == [4]
    assert _status(capsys, lb, home)[0] == 0
    assert calls == [4, 4]


# The judges -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("people", [5, 6])
def test_five_person_scored_runs_read_uncalibrated_and_six_calibrated(people, capsys, tmp_path, home):
    """`calibrated_runs` has 8 person-scored runs; the person's scores of the last 8 - `people` are taken out."""
    folder = tmp_path / "cal" / "lit-brief"
    docs = calibrated_runs(folder)
    scored = [d for d in docs if any((s.get("source") or {}).get("ref") == "person" for s in d["run"]["signals"])]
    for doc in scored[people:]:
        doc["run"]["signals"] = [s for s in doc["run"]["signals"] if (s.get("source") or {}).get("ref") != "person"]
    _stored(home, docs)
    fit = _fit(folder, _store_docs(home, "lit-brief"))[1]
    code, out, err = _status(capsys, folder, home, "--json")
    view = _strict(out)["judges"]
    assert code == 0 and err == "" and view == _view(fit)
    assert list(view) == [*judges.FIT_KEYS, "refused", "anchor"]
    assert all(list(j) == list(judges.JUDGE_KEYS) for j in view["judges"])
    assert (view["calibrated"], view["n_person"]) == (people == 6, people)
    model = [j for j in view["judges"] if j["kind"] == "model"]
    code, out, err = _status(capsys, folder, home)
    lines = out.splitlines()
    assert code == 0 and err == ""
    if people == 5:
        assert all(j["catch_rate"] is None for j in view["judges"])
        assert "  uncalibrated: 5 of 6 person-scored runs; biases are relative to the judges' mean; no catch rates" \
            in lines and not any("catch rate " in line for line in lines)
    else:
        assert model and all(j["catch_rate"] is not None for j in model)
        assert "  calibrated: 6 person-scored runs; biases are relative to the person" in lines
        for j in model:
            [line] = [line for line in lines if line.startswith(f"  {j['id']} (model): ")]
            assert line.endswith(f", catch rate {j['catch_rate']:.2f} of {j['catch_n']}")


def _scores(doc: dict, ref: str | None = None) -> list[dict]:
    return [s for s in doc["run"]["signals"] if s.get("kind") == "score"
            and (ref is None or (s.get("source") or {}).get("ref") == ref)]


def _text(docs, ref, value, picks):
    for doc in picks(docs):
        for s in _scores(doc, ref):
            s["value"] = value


def _nonnumeric(docs):
    _text(docs, "j-claude", "high", lambda ds: [ds[0], ds[10], ds[11]])
    _text(docs, "j-astra", "n/a", lambda ds: [ds[12]])


def _nonfinite(docs):
    _text(docs, "j-astra", 10 ** 400, lambda ds: [ds[1]])


def _only_text(docs):
    _text(docs, None, "good", lambda ds: ds)


def _unreadable(docs):
    docs[2]["run"]["signals"] = {"not": "a list"}


def _unattributed(docs):
    for doc in (docs[0], docs[10], docs[11]):
        for s in _scores(doc):
            s["source"].pop("ref")


def _ledger(base: list[dict], cand: list[dict]) -> dict[str, tuple[int, int]]:
    """Each judge read reason, with the base and candidate runs that carry it."""
    seen: dict[str, list[int]] = {}
    for i, runs in enumerate((base, cand)):
        for doc in runs:
            notes = list(judges.judged(doc, "quality").reasons)
            if unreadable_signals(doc) == SIGNALS_NOT_LIST:
                notes.append("unreadable:signals")
            for w in dict.fromkeys(notes):
                seen.setdefault(w, [0, 0])[i] += 1
    return {w: tuple(n) for w, n in seen.items()}


@pytest.mark.parametrize("change", [_nonnumeric, _nonfinite, _only_text, _unreadable, _unattributed],
                         ids=lambda f: f.__name__.strip("_"))
def test_the_judge_read_lines_count_runs_per_arm_in_compares_words(change, tmp_path):
    """`calibrated_runs`, base against drop-critic, read by status and by compare: each line's words after its
    counts are compare's, and its counts per arm are compare's base and candidate counts."""
    folder = tmp_path / "cal" / "lit-brief"
    docs = calibrated_runs(folder)
    change(docs)
    report = status.read(load(folder), docs)
    base = [d for d in docs if _arm(d) == "base"]
    cand = [d for d in docs if _arm(d) == "drop-critic"]
    lines = [r for r in compare.paired(base, cand, rule="score:quality", now=NOW).reasons
             if "read without" in r or "read with no signals" in r or "name no judge" in r]
    ledger = _ledger(base, cand)
    want = []
    for line in lines:
        if line.startswith("judge scores that name no judge"):
            anon = sorted(w for w in ledger if w.startswith("unattributed:"))
            nb, nc = (sum(ledger[w][i] for w in anon) for i in (0, 1))
            where = " and ".join(f"{n} in {arm} runs" for n, arm in ((nb, "base"), (nc, "candidate")) if n)
            ours = " and ".join(f"{n} in arm {arm} runs" for n, arm in ((nb, "base"), (nc, "drop-critic")) if n)
            assert f"({where})" in line and len(anon) > 5 and line.count("unattributed:") == 5
            want.append(line.replace(f"({where})", f"({ours})"))
            continue
        nb, nc = ledger[line.rsplit(": ", 1)[1]]
        counts = compare.arm_runs(nb, nc, nonzero=True)
        assert line.startswith(counts + " ")
        ours = " and ".join(f"{counted(n, 'run')} of arm {arm}" for n, arm in ((nb, "base"), (nc, "drop-critic")) if n)
        want.append(f"{ours} {line[len(counts) + 1:]}")
    assert lines and report["notes"] == want
    if change is _only_text:
        assert report["judges"] is None and len(lines) == 3
        assert all((r["scored"], r["pooled_mean"]) == (0, None) for r in report["quality"])
    json.dumps(report, allow_nan=False)


# The judge limit warning ------------------------------------------------------------------------------------------

def _many(tmp_path: Path, capsys, n: int, scored: str) -> tuple[Path, Path]:
    """(folder, home): the example flow with `n` declared judges and its plan, and a store of the merge check's
    first 2 pairs (seed 1), each run scored by j-claude and j-astra (`scored` "two") or by every judge."""
    folder, home = tmp_path / f"judges-{n}", tmp_path / f"home-{n}"
    shutil.copytree(EXAMPLE, folder)
    doc = json.loads((folder / "flow.ocp.json").read_text(encoding="utf-8"))
    extra = [f"j{k:02d}" for k in range(3, n)]
    doc["flow"]["judges"] += [dict(doc["flow"]["judges"][0], id=j) for j in extra]
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    (folder / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert _flow(capsys, "plan", str(folder), "--home", str(home))[0] == 0
    flow = load(folder)
    truth = copy.deepcopy(TRUTH)
    truth["judges"].update({j: truth["judges"]["j-claude"] for j in extra})
    specs = [s if scored == "every" else dataclasses.replace(s, judges=("j-claude", "j-astra"), person=False)
             for s in next_runs(flow, n=1000, seed=1) if s.experiment == ARM and int(s.run_id.split(".")[-2]) <= 2]
    specs = [dataclasses.replace(s, person=True) if scored == "every" else s for s in specs]
    _stored(home, simulate.simulate(flow, arm="base", n=0, seed=1, truth=truth, specs=specs))
    return folder, home


VERBS = {"status": lambda folder, home: ["status", str(folder), "--home", str(home)],
         "check": lambda folder, home: ["check", str(folder), "--arm", ARM, "--home", str(home)]}


@pytest.mark.parametrize("scored", ["two", "every"], ids=["14 declared, 2 scored", "14 scored"])
@pytest.mark.parametrize("verb", list(VERBS))
def test_the_judge_warning_is_stderrs_first_line_and_changes_nothing_else(verb, scored, capsys, tmp_path,
                                                                          monkeypatch):
    folder, home = _many(tmp_path, capsys, 14, scored)
    warning = judges.judge_warning(load(folder).flow)
    assert warning is not None and len(load(folder).flow.judges) == 14
    argv = VERBS[verb](folder, home)
    warned = [_flow(capsys, *argv, *more) for more in ((), ("--json",))]
    monkeypatch.setattr(judges, "judge_warning", lambda flow: None)
    quiet = [_flow(capsys, *argv, *more) for more in ((), ("--json",))]
    for (code, out, err), (q_code, q_out, q_err) in zip(warned, quiet):
        assert (code, out) == (q_code, q_out) and code == 0 and out
        assert err == f"warning: {warning}\n" and q_err == ""
    report = _strict(warned[1][1])
    if verb == "status":
        assert report["judges"]["refused"] == ("too-many-judges:14" if scored == "every" else None)
    else:
        assert report["schema"] == "loopmath.flow-check/1" and report["arm"] == ARM


def test_thirteen_judges_warn_nothing_and_a_flow_that_does_not_load_gives_its_refusal_only(capsys, tmp_path):
    folder, home = _many(tmp_path, capsys, 13, "every")
    assert judges.judge_warning(load(folder).flow) is None
    for verb in VERBS.values():
        code, out, err = _flow(capsys, *verb(folder, home), "--json")
        assert code == 0 and err == "" and _strict(out)
    assert _strict(_flow(capsys, *VERBS["status"](folder, home), "--json")[1])["judges"]["refused"] is None
    broken, _ = _many(tmp_path / "broken", capsys, 14, "two")
    (broken / "flow.ocp.json").write_text("{", encoding="utf-8")
    for verb in VERBS.values():
        code, out, err = _flow(capsys, *verb(broken, home))
        assert code == 1 and out == "" and err.startswith("error: ") and "warning" not in err


# Without a usable plan --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["no plan", "stale", "another version"])
def test_without_a_usable_plan_status_names_why_exits_0_and_still_reads_the_runs(which, capsys, lb, home):
    _stored(home, _screen(lb, 4))
    plan = lb / "plan.json"
    held = json.loads(plan.read_text(encoding="utf-8"))
    if which == "no plan":
        plan.unlink()
    else:
        held["flow"].update({"digest": "flw_000000000000"} if which == "stale" else {"version": 2})
        plan.write_text(json.dumps(held), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _usable_plan(load(lb))
    problem = str(caught.value)
    if which == "no plan":
        assert problem == f"there is no plan.json in {lb}; run loopmath flow plan first"
    before = _files(lb, home)
    code, out, err = _status(capsys, lb, home, "--json")
    report = _strict(out)
    assert code == 0 and err == "" and list(report) == list(status.STATUS_KEYS)
    assert (report["plan"], report["plan_problem"]) == (None, problem)
    assert (report["experiments"], report["wait"], report["adopted"]) == ([], None, None)
    assert report["runs"] == 4 and report["cost"] and report["checks"] and report["judges"] is not None
    assert [r["runs"] for r in report["quality"]] == [1, 1, 1, 1] and report["dollars"]["runs"] == 4
    code, out, err = _status(capsys, lb, home)
    lines = out.splitlines()
    assert code == 0 and err == "" and lines[1] == f"plan: {problem}"
    assert lines[lines.index("experiments:") + 1].startswith("dollars: ")
    assert not any(line in ("wait:",) or line.startswith("adopted:") for line in lines)
    assert _files(lb, home) == before


# Dollars ----------------------------------------------------------------------------------------------------------

def _attempt(step: str | None, model: dict | None, usd=None, *, cost: bool = True) -> dict:
    a = {"id": f"att_{step}_{usd!r:.12}", "status": "done"}
    if step is not None:
        a["node"] = step
    if model is not None:
        a["model"] = model
    if cost:
        a["cost"] = {"usd": usd, "basis": "reported"}
    return a


SONNET, OPUS, HAIKU = ({"raw": m, "id": m} for m in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"))


def test_each_attempt_is_read_once_for_the_rows_and_the_dollars_and_the_known_part_is_kept(lb):
    """Dyadic dollars across steps and models, a bool, a missing cost, numbers past a float (10**400 either sign,
    NaN), an attempt that is not an object, a step and model the attempt does not name, and a run with none."""
    a, b, c, d = copy.deepcopy(_screen(lb, 4))
    a["attempts"] = [_attempt("extract", SONNET, 0.5), _attempt("synthesize", OPUS, 0.25), 7,
                     _attempt(None, {"raw": "custom-model"}, 0.03125)]
    b["attempts"] = [_attempt("extract", SONNET, 0.125), _attempt("synthesize", OPUS, cost=False),
                     _attempt("synthesize", OPUS, True)]
    c["attempts"] = [_attempt("scout", HAIKU, 10 ** 400), _attempt("scout", HAIKU, -(10 ** 400)),
                     _attempt("scout", HAIKU, float("nan")), _attempt("scout", HAIKU, 0.0625)]
    d["attempts"] = []
    report = status.read(load(lb), [a, b, c, d])
    assert report["cost"] == [dict(zip(status.COST_KEYS, row)) for row in [
        ("scout", "claude-haiku-5-5", 4, 1, 0.0625, None, "3 of 4 attempts have dollars past what a float holds"),
        ("extract", "claude-sonnet-5-5", 2, 2, 0.625, 0.625, None),
        ("synthesize", "claude-opus-5-5", 3, 1, 0.25, None, "2 of 3 attempts have no dollars"),
        ("unknown", "custom-model", 1, 1, 0.03125, 0.03125, None)]]
    assert report["dollars"] == dict(zip(status.DOLLAR_KEYS, (
        4, 1, 10, 5, 0.96875, None, "1 of 4 runs has an attempt with no dollars; 1 of 4 runs has an attempt with "
        "dollars past what a float holds; 1 of 4 runs has no attempts")))
    assert math.fsum(r["usd_known"] for r in report["cost"]) == report["dollars"]["usd_known"]
    assert report["notes"] == [f"cost:{c['run']['id']}"]
    json.dumps(report, allow_nan=False)


def test_a_sum_past_a_float_is_unknown_with_its_reason_in_standard_json(capsys, lb, home):
    """Two attempts of 1e308 at one step and model of one run: its row, the run and the total are past a float."""
    docs = _screen(lb, 4)
    scout = [x for x in docs[0]["attempts"] if x["node"] == "scout"]
    assert len(scout) == 2 and len({x["model"]["id"] for x in scout}) == 1
    for x in scout:
        x["cost"]["usd"] = 1e308
    _stored(home, docs)
    code, out, err = _status(capsys, lb, home, "--json")
    report = _strict(out)
    assert code == 0 and err == ""
    model = scout[0]["model"]["id"]
    [row] = [r for r in report["cost"] if (r["step"], r["model"]) == ("scout", model)]
    assert (row["usd_known"], row["usd"], row["reason"]) == (None, None, status.SUM_PAST)
    assert all(r["usd"] is not None for r in report["cost"] if r is not row)
    assert report["dollars"] == {"runs": 4, "priced_runs": 4, "attempts": sum(len(x["attempts"]) for x in docs),
                                 "priced": sum(len(x["attempts"]) for x in docs), "usd_known": None, "usd": None,
                                 "reason": "the sum is past what a float holds"}
    assert report["notes"] == [f"cost:{docs[0]['run']['id']}"]
    code, out, err = _status(capsys, lb, home)
    lines = out.splitlines()
    assert code == 0 and err == ""
    assert f"  scout, {model}: {counted(row['attempts'], 'attempt')}, the sum is past what a float holds" in lines
    assert "dollars: 4 runs, 4 priced; the sum is past what a float holds" in lines
    assert lines[-2:] == ["notes:", f"  cost:{docs[0]['run']['id']}"]


# Arms and checks --------------------------------------------------------------------------------------------------

def test_arms_are_the_plans_in_its_order_then_any_other_sorted(capsys, lb, home, tmp_path):
    """A run `flow record --allow-unplanned` records under arm `unplanned` (a row with a configuration of its own,
    `_own`), and a stored run under an arm the plan does not name, `aa-other`: their checks are their stored
    configurations' checks."""
    docs = _screen(lb, 4)
    docs[2]["run"]["flow"]["arm"] = "aa-other"
    _stored(home, docs)
    own, path = _own(lb), tmp_path / "unplanned.json"
    path.write_text(json.dumps(own), encoding="utf-8")
    code, out, err = _flow(capsys, "record", str(lb), "--ocp", str(path), "--allow-unplanned", "--home", str(home))
    assert code == 0 and " arm unplanned " in out, err
    planned = [a for a in _arm_order(lb, docs) if a != "aa-other"]
    assert len(planned) == 3
    report = _strict(_status(capsys, lb, home, "--json")[1])
    assert [r["arm"] for r in report["quality"]] == planned + ["aa-other", "unplanned"]
    assert [r["runs"] for r in report["quality"]] == [1, 1, 1, 1, 1]
    signals = Store(home).run_doc(own["run"]["id"])["run"]["signals"]
    held = any(s.get("kind") == "verdict" and s.get("name") == "claims-check" for s in signals)
    [row] = [r for r in report["checks"] if (r["check"], r["arm"]) == ("claims-check", "unplanned")]
    assert (row["runs"], row["missing"]) == (1, 0 if held else 1)
    claims = [r["arm"] for r in report["checks"] if r["check"] == "claims-check"]
    assert claims == planned + ["aa-other", "unplanned"]


# A workflow the catalog does not hold -----------------------------------------------------------------------------

UNKNOWN = {"ref": "workflow-not-in-this-catalog", "version": 1}


def _record_unplanned(capsys, lb: Path, home: Path, tmp_path: Path, own: dict) -> None:
    """`own`, strict valid, recorded with `flow record --allow-unplanned`."""
    assert [f for f in validate_strict(own) if f["level"] == "error"] == []
    path = tmp_path / "unplanned.json"
    path.write_text(json.dumps(own), encoding="utf-8")
    code, out, err = _flow(capsys, "record", str(lb), "--ocp", str(path), "--allow-unplanned", "--home", str(home))
    assert code == 0 and " arm unplanned " in out, err


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_recorded_run_whose_workflow_the_catalog_does_not_hold_is_refused_by_name(as_json, capsys, lb, home,
                                                                                    tmp_path):
    """A strict-valid run of arm `unplanned` whose configuration names a workflow this catalog does not hold (one
    made where that workflow is installed): `flow record` stores it, and `flow status` cannot name its checks, so
    it refuses by name and writes nothing, never reading the run as one with no checks."""
    _stored(home, _screen(lb, 4))
    own = _own(lb)
    own["run"]["configuration"]["workflow"] = dict(UNKNOWN)
    _record_unplanned(capsys, lb, home, tmp_path, own)
    reason = (f"run {own['run']['id']}'s workflow workflow-not-in-this-catalog version 1 is not in the catalog, so "
              "flow status cannot name its checks")
    before = _files(lb, home)
    code, out, err = _status(capsys, lb, home, *(["--json"] if as_json else []))
    if as_json:
        assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow status",
                                                 "error": reason, "exit": 1}
    else:
        assert code == 1 and out == "" and err == f"error: {reason}\n"
    assert _files(lb, home) == before


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_flow_whose_workflow_the_catalog_does_not_hold_is_refused_by_name(as_json, capsys, lb, home):
    """The flow's own configuration naming a workflow this catalog does not hold: the flow loads, and `flow
    status` cannot list its steps, so it refuses by name, as `flow show` does."""
    path = lb / "flow.ocp.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["flow"]["configuration"]["workflow"] = dict(UNKNOWN)
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert load(lb).flow.configuration["workflow"] == UNKNOWN
    reason = ("the flow's workflow workflow-not-in-this-catalog version 1 is not in the catalog, so flow status "
              "cannot list its steps")
    before = _files(lb, home)
    code, out, err = _status(capsys, lb, home, *(["--json"] if as_json else []))
    if as_json:
        assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow status",
                                                 "error": reason, "exit": 1}
    else:
        assert code == 1 and out == "" and err == f"error: {reason}\n"
    assert _files(lb, home) == before


def test_a_recorded_run_of_a_catalog_workflow_shows_that_workflows_checks(capsys, lb, home, tmp_path):
    """The control: a run whose configuration names `implement_review` from the catalog (each piece given the
    run's first setting) shows that workflow's check, `review`, beside the rule's `claims-check`. The inline
    configurations are `test_arms_are_the_plans_in_its_order_then_any_other_sorted`'s."""
    own = _own(lb)
    workflow = catalog()["implement_review"]
    setting = next(iter(own["run"]["configuration"]["settings"].values()))
    configuration = {"workflow": {"ref": "implement_review", "version": workflow.version},
                     "settings": {p.id: setting for p in workflow.pieces}}
    configuration["id"] = config_id(configuration["workflow"], configuration["settings"],
                                    resolve=conformance.catalog_resolver)
    own["run"]["configuration"] = configuration
    _record_unplanned(capsys, lb, home, tmp_path, own)
    code, out, err = _status(capsys, lb, home, "--json")
    assert code == 0 and err == ""
    assert [(r["check"], r["arm"], r["runs"], r["missing"]) for r in _strict(out)["checks"]] == [
        ("claims-check", "unplanned", 1, 1), ("review", "unplanned", 1, 1)]


@pytest.mark.parametrize("resolver, raised", [(lambda ref, version: None, status.UnresolvedWorkflow),
                                              (lambda ref, version: {}["x"], KeyError),
                                              (lambda ref, version: [][0], IndexError)],
                         ids=["not in the catalog", "a KeyError", "an IndexError"])
def test_only_the_resolvers_own_lookup_error_is_a_workflow_not_in_the_catalog(resolver, raised, monkeypatch):
    """The resolver finding nothing raises the plain LookupError that status names; a KeyError or IndexError
    from resolving is a bug and is raised as it is."""
    monkeypatch.setattr(workflows_ocp, "catalog_resolver", resolver)
    with pytest.raises(raised) as caught:
        status._resolved({"workflow": {"ref": "solo", "version": 1}}, "the flow's", "list its steps")
    assert type(caught.value) is raised
    if raised is status.UnresolvedWorkflow:
        assert str(caught.value) == ("the flow's workflow solo version 1 is not in the catalog, so flow status "
                                     "cannot list its steps")


def _verdict(template: dict, n: int, value: str, at: str) -> dict:
    return dict(template, id=f"{template['id']}-{n}", value=value, observed_at=at)


def test_a_check_counts_each_run_once_by_its_newest_verdict(lb):
    """Six base runs (seed 1), their claims-check verdicts set: newest by instant over the text's order, a full
    tie read in document order (the last), an unknown value, none, an unread time read oldest, and upper case."""
    docs = simulate.simulate(load(lb), arm="base", n=6, seed=1, truth=TRUTH)
    plans = [[("pass", "2026-11-03T12:00:00-08:00"), ("fail", "2026-11-03T19:00:00Z")],
             [("pass", "2026-11-03T20:00:00Z"), ("fail", "2026-11-03T12:00:00-08:00")],
             [("flaky", "2026-11-03T20:00:00Z")],
             [],
             [("FAIL", "2026-11-03T20:00:00Z"), ("Pass", "not a time")],
             [("PASS", "2026-11-03T20:00:00Z")]]
    for doc, given in zip(docs, plans):
        signals = doc["run"]["signals"]
        held = [s for s in signals if s.get("kind") == "verdict" and s.get("name") == "claims-check"]
        doc["run"]["signals"] = [s for s in signals if s not in held] + [
            _verdict(held[0], n, value, at) for n, (value, at) in enumerate(given)]
    docs[0]["run"]["signals"].append(_verdict(next(s for s in docs[0]["run"]["signals"] if s["kind"] == "verdict"),
                                              9, "fail", "2026-11-04T00:00:00Z") | {"name": "accepted"})
    report = status.read(load(lb), docs)
    [row] = [r for r in report["checks"] if r["check"] == "claims-check"]
    assert row == {"check": "claims-check", "arm": "base", "runs": 6, "pass": 2, "fail": 2, "other": 1, "missing": 1,
                   "rate": 0.5}
    assert "accepted" not in {r["check"] for r in report["checks"]}


def test_a_rule_that_names_no_requires_has_the_tests_check(capsys, tmp_path, home):
    """The converter's default verdict name, `tests`, is the rule's check when it names no `requires`."""
    folder = tmp_path / "no-requires"
    shutil.copytree(EXAMPLE, folder, ignore=shutil.ignore_patterns("flow.toml"))
    path = folder / "flow.ocp.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["flow"]["acceptance_rule"]["requires"]
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert _flow(capsys, "plan", str(folder), "--home", str(home))[0] == 0
    docs = _screen(folder, 4)
    assert not [s for d in docs for s in d["run"]["signals"] if s.get("name") == "tests"]
    _stored(home, docs)
    code, out, err = _status(capsys, folder, home, "--json")
    rows = [r for r in json.loads(out)["checks"] if r["check"] == "tests"]
    assert code == 0 and rows and sum(r["runs"] for r in rows) == 4
    assert all(r["missing"] == r["runs"] and r["pass"] == r["fail"] == r["other"] == 0 for r in rows)


def test_runs_of_another_revision_of_the_flow_are_counted_apart(capsys, lb, home):
    _stored(home, _screen(lb, 2))
    path = lb / "flow.ocp.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["flow"]["version"] = 2  # the digest leaves the version out
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    code, out, err = _status(capsys, lb, home)
    assert code == 0 and "runs: 0; 2 of other revisions of flow lit-brief" in out.splitlines()
    code, out, err = _status(capsys, lb, home, "--json")
    report = json.loads(out)
    assert code == 0 and (report["runs"], report["other_revisions"]) == (0, 2)
