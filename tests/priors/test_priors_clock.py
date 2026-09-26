"""No real date or clock time ships, from any source (22L for lanes, 23B for sweep, e0 and rq1).

Every converter's output, after the share reduction, and every shipped file (the
runs and the manifest) are walked key by key and value by value, as 22L's lanes
tests do: no source date (either `YYYY-MM-DD` or `YYYYMMDD`), no UTC offset,
every time-shaped string on the synthetic clock (`1970-...Z`), and no number
within a year of a source time read as Unix seconds, milliseconds or
microseconds. The price table's as-of date (`cost.tariff.date`, public, one per
table) is the only date a run keeps; the manifest also has `built_at`, the build's
UTC date.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from loopmath.belief.design import parse_run
from loopmath.priors import manifest, ocpdoc
from loopmath.priors.e0 import iter_e0
from loopmath.priors.lanes import convert_lane
from loopmath.priors.reduce import reduce_for_bundle
from loopmath.priors.rq1 import normalize
from loopmath.priors.sweep import convert_sweep_run

from test_priors_bundle import SHIPPED  # noqa: E402  (same folder)
from test_priors_lanes import _row
from test_priors_sources import _session, lane10_doc
from test_priors_sweep import accepted_run, rows_for

TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d")
DATE = re.compile(r"(19|20)\d\d-[01]\d-[0-3]\d")
COMPACT = re.compile(r"(?<![0-9A-Za-z])(19|20)\d\d[01]\d[0-3]\d(?![0-9A-Za-z])")
OFFSET = re.compile(r"[+-]\d\d:\d\d")
SYNTHETIC = re.compile(r"^1970-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
YEAR = 366 * 86400
TIME_KEYS = {"started_at", "ended_at", "observed_at", "emitted_at"}


def _walk(obj, path=()):
    """(key path, scalar) for every value, but the price table's as-of date."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not (k == "date" and path[-1:] == ("tariff",)):
                yield from _walk(v, path + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, path)
    else:
        yield path, obj


def _source_times(obj) -> list[datetime]:
    out = []
    for _, v in _walk(obj):
        if isinstance(v, str) and TIME.match(v):
            out.append(datetime.fromisoformat(v.replace("Z", "+00:00")))
    return out


def _assert_no_source_time(docs, source):
    times = _source_times(source)
    assert times  # the fixture has real times to lose
    dates = {t.strftime(f) for t in times for f in ("%Y-%m-%d", "%Y%m%d")}
    dates |= {t.astimezone(timezone.utc).strftime(f) for t in times for f in ("%Y-%m-%d", "%Y%m%d")}
    stamps = [t.timestamp() for t in times]
    for path, leaf in _walk(docs):
        if isinstance(leaf, str):
            assert not any(d in leaf for d in dates), (path, leaf)
            assert not OFFSET.search(leaf) and not COMPACT.search(leaf), (path, leaf)
            if DATE.search(leaf):
                assert SYNTHETIC.match(leaf), (path, leaf)
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            for scale in (1, 1e3, 1e6):
                assert all(abs(leaf / scale - t) > YEAR for t in stamps), (path, leaf)


def _local(src):
    """The same source with every UTC time written in local time (-07:00): the offset must not ship either."""
    text = json.dumps(src)
    for t in sorted(set(re.findall(r'"(\d{4}-\d\d-\d\dT[^"]*?)"', text))):
        local = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone(-timedelta(hours=7)))
        text = text.replace(f'"{t}"', f'"{local.isoformat()}"')
    return json.loads(text)


def _sweep():
    src = _local(accepted_run())
    doc, _ = convert_sweep_run(src, rows_for(src), stem="t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh")
    return src, [doc]


def _e0(tmp_path):
    main = _session("main-1")
    main.update(started_at="2026-08-20T10:00:00.250-07:00", ended_at="2026-08-20T11:00:00.750-07:00")
    late = _session("agent-late", model="claude-fable-5", extras={"parent_session_id": "main-1"})
    late.update(started_at="2026-08-20T10:30:00-07:00", ended_at="2026-08-20T10:40:00-07:00")
    early = _session("agent-early", model="claude-sonnet-5", extras={"parent_session_id": "main-1"})
    early.update(started_at="2026-08-20T10:10:00-07:00", ended_at="2026-08-20T10:20:00-07:00")
    sessions = [main, late, early]  # the file lists the later child first
    (tmp_path / "sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sessions))
    return sessions, list(iter_e0(tmp_path))


def _rq1():
    src = lane10_doc()
    run = src["run"]
    run["id"] = "rq1-ahc039-A01-20260102-030405-phase1"
    run["signals"][0]["id"] = f"sig_{run['id']}_heldout_perf"
    run["ext"]["dev.loopmath.rq1"]["run_id"] = run["id"]
    run.update(started_at="2026-01-02T03:04:05-07:00", ended_at="2026-01-02T05:04:05-07:00")
    run["signals"][0]["observed_at"] = "2026-01-02T05:05:05-07:00"
    src["attempts"][0].update(started_at="2026-01-02T03:04:10-07:00", ended_at="2026-01-02T03:09:05-07:00")
    src["producer"]["emitted_at"] = "2026-01-02T17:51:03-07:00"
    return src, [normalize(src)]


def _lanes():
    rows = [_row("s8"), _row("s9", verdicts=())]
    return rows, [convert_lane(r) for r in rows]


def _converted(source, tmp_path):
    return {"sweep": _sweep, "e0": lambda: _e0(tmp_path), "rq1": _rq1, "lanes": _lanes}[source]()


@pytest.mark.parametrize("source", ["sweep", "e0", "rq1", "lanes"])
def test_no_source_time_survives_reduction(source, tmp_path):
    src, docs = _converted(source, tmp_path)
    reduced = [reduce_for_bundle(d, salt="x") for d in docs]
    _assert_no_source_time(reduced, src)
    for d in reduced:
        assert d["run"]["started_at"] == ocpdoc.EPOCH_TEXT
        assert d["run"]["ext"]["dev.loopmath.prior"]["clock"] == ocpdoc.CLOCK


def test_sweep_elapsed_seconds_and_real_order_survive(tmp_path):
    _, [doc] = _sweep()
    assert [(a["id"], a["started_at"], a["ended_at"]) for a in doc["attempts"]] == [
        ("plan.a1", "1970-01-01T00:00:00Z", "1970-01-01T00:01:00Z"),
        ("implement.a1", "1970-01-01T00:10:00Z", "1970-01-01T00:12:00Z"),
        ("tests.a1", "1970-01-01T00:12:00Z", "1970-01-01T00:12:01Z"),
        ("implement.a2", "1970-01-01T00:12:01Z", "1970-01-01T00:14:00Z"),
        ("tests.a2", "1970-01-01T00:14:00Z", "1970-01-01T00:14:01Z"),
        ("review.a1", "1970-01-01T00:14:01Z", "1970-01-01T00:15:00Z")]
    assert doc["attempts"][3]["cause"] == {"type": "gate_failed", "ref": "tests.a1"}  # interleaved by real time
    assert {s["name"]: s["observed_at"] for s in doc["run"]["signals"]} == {
        "tests": "1970-01-01T00:14:01Z", "tests_pass_fraction": "1970-01-01T00:14:01Z",
        "referee": "1970-01-01T00:15:00Z"}
    assert (doc["run"]["ended_at"], doc["producer"]["emitted_at"]) == ("1970-01-01T00:15:00Z", "1970-01-01T01:00:00Z")
    assert parse_run(doc).evidence.z == 1.0


def test_e0_children_keep_their_real_order(tmp_path):
    _, [doc] = _e0(tmp_path)
    assert [(a["model"]["id"], a["started_at"], a["ended_at"]) for a in doc["attempts"]] == [
        ("opus-5", "1970-01-01T00:00:00Z", "1970-01-01T01:00:00Z"),  # the main session, rounded to whole seconds
        ("sonnet-5", "1970-01-01T00:10:00Z", "1970-01-01T00:20:00Z"),
        ("fable-5", "1970-01-01T00:30:00Z", "1970-01-01T00:40:00Z")]
    assert doc["run"]["task"]["source"] == {"kind": "e0", "ref": "e0-corpus"}


def test_rq1_ids_lose_the_start_stamp():
    _, [doc] = _rq1()
    run = doc["run"]
    assert run["id"] == "rq1-ahc039-A01-phase1" and run["signals"][0]["id"] == "sig_rq1-ahc039-A01-phase1_heldout_perf"
    assert (run["started_at"], run["ended_at"], run["signals"][0]["observed_at"]) == (
        "1970-01-01T00:00:00Z", "1970-01-01T02:00:00Z", "1970-01-01T02:01:00Z")
    assert doc["attempts"][0]["started_at"] == "1970-01-01T00:00:05Z"


# ---------------------------------------------------------------- the shipped files
def _shipped():
    m = manifest(SHIPPED)
    if not m.get("sources"):
        pytest.skip("no bundle built in this checkout")
    docs = {}
    for name, entry in m["sources"].items():
        with gzip.open(SHIPPED / entry["file"], "rt", encoding="utf-8") as fh:
            docs[name] = [json.loads(line) for line in fh if line.strip()]
    return m, docs


# Our runs' real times fall in 2026; a Unix-seconds number within a year of that range is a leak.
_LO = datetime(2025, 8, 1, tzinfo=timezone.utc).timestamp()
_HI = datetime(2027, 10, 1, tzinfo=timezone.utc).timestamp()


def _assert_clean(obj, *, allow=()):
    for path, leaf in _walk(obj):
        if isinstance(leaf, str):
            if path in allow:
                continue
            assert not COMPACT.search(leaf), (path, leaf)
            assert all(m.group().startswith("1970-") for m in DATE.finditer(leaf)), (path, leaf)  # notes name the epoch
            if path[-1:] and path[-1] in TIME_KEYS:
                assert SYNTHETIC.match(leaf), (path, leaf)
            assert not (TIME.search(leaf) and OFFSET.search(leaf)), (path, leaf)
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool) and not str(path[-1]).endswith("tokens"):
            for scale in (1, 1e3, 1e6):
                assert not _LO <= leaf / scale <= _HI, (path, leaf)


def test_the_shipped_runs_carry_no_real_time():
    _, docs = _shipped()
    assert set(docs) >= {"sweep", "e0", "rq1", "lanes"}
    for name, runs in docs.items():
        _assert_clean(runs)
        for d in runs:
            assert d["run"]["started_at"] == ocpdoc.EPOCH_TEXT, (name, d["run"]["id"])
            assert d["run"]["ext"]["dev.loopmath.prior"]["clock"] == ocpdoc.CLOCK, (name, d["run"]["id"])


def test_the_shipped_manifest_carries_the_build_date_only():
    m, _ = _shipped()
    assert re.fullmatch(r"\d{4}-\d\d-\d\d", m["built_at"])
    _assert_clean(m, allow={("built_at",)})
