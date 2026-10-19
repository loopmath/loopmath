"""The generic run log `loopmath.harness-runs/0.1`: the row checks, the file reader and writer, and the
published schema (spec/harness-runs.md, spec/harness-runs-0.1.schema.json)."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import jsonschema
import pytest

from loopmath import harness_runs as HR
from loopmath.harness_runs import rows as rows_mod
from loopmath.types import TASK_TYPE_IDS

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "spec" / "harness-runs-0.1.schema.json"
EXAMPLES = ROOT / "spec" / "examples" / "harness-runs"


def _good(**over) -> dict:
    row = {"schema": "loopmath.harness-runs/0.1", "task_id": "ledger-1", "arm": "base", "agent": "opencode",
           "model": "gemini-3.5-flash", "effort": None,
           "outcome": {"name": "judge", "score": 0.75, "judge": "judge-x", "scale": "fraction"},
           "cost": {"usd": 0.42, "tokens": {"input": 10, "cache_read": 0, "cache_write": 0, "output": 5},
                    "recovered": True, "usd_source": "opencode"},
           "started_at": "2026-10-18T17:10:00-07:00", "ended_at": "2026-10-18T17:14:02-07:00",
           "tool": {"name": "harness2", "run_id": "jb-oc-agg2/t1/base1"}, "ext": {"org.example": {"a": 1}}}
    row.update(over)
    return row


def test_schema_constant():
    assert HR.SCHEMA == "loopmath.harness-runs/0.1"


def test_a_minimal_row_needs_only_schema_and_task_id():
    assert HR.check_row({"schema": HR.SCHEMA, "task_id": "t"}) == []
    assert HR.check_row(_good()) == []


def test_row_builder_sets_the_schema_and_checks():
    row = HR.row("t1", model="m", outcome={"score": None, "judge": "j"})
    assert row["schema"] == HR.SCHEMA and row["task_id"] == "t1"
    with pytest.raises(ValueError, match="task_id"):
        HR.row("")


BAD = [
    ({"task_id": None}, "task_id"),
    ({"task_id": ""}, "task_id"),
    ({"schema": "loopmath.harness-runs/0.2"}, "loopmath.harness-runs/0.2"),
    ({"schema": "harness-runs"}, "harness-runs"),
    ({"outcome": {"score": "0.5", "judge": "j"}}, "outcome.score"),
    ({"outcome": {"judge": "j"}}, "outcome"),                       # neither score nor accepted
    ({"outcome": {"score": 1, "accepted": True}}, "outcome"),      # both
    ({"outcome": {"accepted": "yes"}}, "outcome.accepted"),
    ({"outcome": {"score": 1, "scale": "percent"}}, "outcome.scale"),
    ({"outcome": {"score": 1, "better": "up"}}, "outcome.better"),
    ({"outcome": {"score": 1.5, "judge": "j", "scale": "fraction"}}, "outcome.score"),   # a fraction is in [0, 1]
    ({"outcome": {"score": -0.1, "judge": "j", "scale": "fraction"}}, "outcome.score"),
    ({"cost": {"usd": -1}}, "cost.usd"),
    ({"cost": {"tokens": {"input": -3}}}, "cost.tokens.input"),
    ({"cost": {"tokens": {"prompt": 3}}}, "cost.tokens.prompt"),
    ({"cost": {"tokens": 18000}}, "cost.tokens"),
    ({"cost": {"recovered": "true"}}, "cost.recovered"),
    ({"started_at": "2026-10-18 17:10"}, "started_at"),
    ({"ended_at": "2026-10-18T17:10:00"}, "ended_at"),            # no offset
    ({"harness": 5}, "harness"),
    ({"harness": {"digest": "sha256:" + "0" * 64}}, "harness"),    # a record needs its schema and factors
    ({"tool": {"name": 3}}, "tool.name"),
    ({"ext": []}, "ext"),
    ({"arm": 1}, "arm"),
    ({"pair": 2}, "pair"),
    ({"experiment": 7}, "experiment"),
    ({"status": "crashed"}, "status"),
    ({"cost": {"complete": "no"}}, "cost.complete"),
    ({"cost": {"usd": 0.01, "complete": False}}, "cost.usd"),     # a partial sum is never usd
    ({"cost": {"partial_usd": -0.5}}, "cost.partial_usd"),
    ({"cost": {"incomplete": 3}}, "cost.incomplete"),
    ({"task": {"type": "bugfix"}}, "task.type"),                  # not a loopmath task type
    ({"task": {"type": ""}}, "task.type"),
    ({"experiment": "/Users/someone/runs/exp1"}, "experiment"),   # never a path
    ({"experiment": "runs/exp1"}, "experiment"),
    ({"experiment": "~exp"}, "experiment"),
    ({"experiment": "..\\exp1"}, "experiment"),
    ({"experiment": ".."}, "experiment"),
]


@pytest.mark.parametrize("over,named", BAD, ids=[n for _, n in BAD])
def test_a_bad_row_is_refused_by_name(over, named):
    problems = HR.check_row(_good(**over))
    assert problems and any(named in p for p in problems), problems


@pytest.mark.parametrize("over,named", BAD, ids=[n for _, n in BAD])
def test_the_published_schema_refuses_the_same_rows(over, named):
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(_good()))
    assert list(validator.iter_errors(_good(**over)))


def test_an_unknown_task_type_names_the_types_and_the_closest_one():
    problem = next(p for p in HR.check_row(_good(task={"type": "bugfix"})) if p.startswith("task.type"))
    assert "'bugfix'" in problem and "bug_fix, feature" in problem and "did you mean 'bug_fix'?" in problem
    assert "did you mean 'refactor'?" in HR.check_row(_good(task={"type": "Re-factor"}))[0]
    assert "did you mean" not in HR.check_row(_good(task={"type": "zzz"}))[0]
    for name in TASK_TYPE_IDS:
        assert HR.check_row(_good(task={"type": name})) == [], name
    assert HR.check_row(_good(task={"type": None, "repo": "acme/x"})) == []


def test_an_experiment_that_is_a_path_is_refused_and_a_dotted_name_is_kept():
    assert "is a path" in HR.check_row(_good(experiment="/Users/someone/runs/exp1"))[0]
    for name in ("exp1", "lint-plugin-eval-1", "h2-0123456789abcdef", "v1.2", ".hidden", "a..b"):
        assert HR.check_row(_good(experiment=name)) == [], name


def test_a_row_without_schema_says_it_is_missing():
    row = _good()
    del row["schema"]
    assert HR.check_row(row) == ["schema: missing (this reader reads loopmath.harness-runs/0.1)"]
    assert HR.check_row(_good(schema=None))[0].startswith("schema: missing")


def test_a_file_with_a_byte_order_mark_is_refused_in_plain_words(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(_good()) + "\n").encode("utf-8"))
    with pytest.raises(ValueError) as got:
        HR.read(path)
    assert str(got.value) == f"{path}:1: the file starts with a byte order mark; save it as UTF-8 without one"


def test_read_numbered_gives_each_row_its_line(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n" + json.dumps(_good(task_id="a")) + "\n  \n" + json.dumps(_good(task_id="b")) + "\n",
                    encoding="utf-8")
    assert [(n, r["task_id"]) for n, r in rows_mod.read_numbered(path)] == [(2, "a"), (4, "b")]


def test_only_a_fraction_is_bounded():
    for scale, score in (("linear", 1.5), ("log", -2.0), ("fraction", 0.0), ("fraction", 1), ("fraction", None)):
        assert HR.check_row(_good(outcome={"score": score, "judge": "j", "scale": scale})) == [], (scale, score)


def test_unknown_fields_are_kept_not_refused():
    row = _good(stage_notes="kept", ext={"x.y": {"z": 1}})
    assert HR.check_row(row) == []


def test_read_refuses_the_file_on_the_first_bad_line_by_number(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(_good()) + "\n\n" + json.dumps(_good()) + "\n" + "{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"runs\.jsonl:4"):
        HR.read(path)
    path.write_text(json.dumps(_good()) + "\n" + json.dumps(_good(schema="x/9")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"runs\.jsonl:2: .*x/9"):
        HR.read(path)


def test_append_then_read_round_trips_and_skips_blank_lines(tmp_path):
    path = tmp_path / "log" / "runs.jsonl"
    path.parent.mkdir()
    HR.append(path, _good())
    HR.append(path, _good(task_id="ledger-2", outcome=None))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(x)["schema"] == HR.SCHEMA for x in lines)
    assert [r["task_id"] for r in HR.read(path)] == ["ledger-1", "ledger-2"]
    with pytest.raises(ValueError, match="task_id"):
        HR.append(path, _good(task_id=""))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


SEPARATORS = "one\u2028two\u2029three\u0085four\x1cfive\x0bsix\x0cseven"   # each one str.splitlines splits on


def test_a_row_whose_text_holds_line_separators_reads_back_as_append_wrote_it(tmp_path):
    path = tmp_path / "runs.jsonl"
    rows = [_good(task={"title": SEPARATORS}), _good(task_id="ledger-2", ext={"org.example": {"note": SEPARATORS}})]
    for r in rows:
        HR.append(path, r)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and "\u2028".encode() in raw and "\u0085".encode() in raw   # written raw, not escaped
    assert HR.read(path) == rows


@pytest.mark.parametrize("step", [1, 40, 0, OSError(28, "No space left on device")],
                         ids=["one-byte", "short", "zero", "error"])
def test_a_write_that_stops_short_is_cut_back_and_raised_never_left_torn(tmp_path, monkeypatch, step):
    path = tmp_path / "runs.jsonl"
    HR.append(path, _good())
    before = path.read_bytes()
    real = os.write

    def short(fd, data):     # os.write as a fake: it writes only `step` bytes, or raises
        if isinstance(step, OSError):
            raise step
        return real(fd, bytes(data[:step])) if step else 0
    monkeypatch.setattr(os, "write", short)
    with pytest.raises(OSError, match="No space" if isinstance(step, OSError) else f"wrote {step} of"):
        HR.append(path, _good(task_id="ledger-2"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    HR.append(path, _good(task_id="ledger-3"))
    assert [r["task_id"] for r in HR.read(path)] == ["ledger-1", "ledger-3"]


def _in_thread(target, *args) -> threading.Thread:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def test_a_cut_back_never_takes_a_row_another_writer_appended(tmp_path, monkeypatch):
    """Writer A's fsync fails; writer B appends between A's size check and A's cut. B's row stays."""
    path = tmp_path / "runs.jsonl"
    HR.append(path, _good())
    a, b = threading.get_ident(), {}
    real_fsync, real_truncate = os.fsync, os.ftruncate

    def fsync(fd):
        if threading.get_ident() == a:
            raise OSError(errno.EIO, "Input/output error")
        return real_fsync(fd)

    def writer_b():
        HR.append(path, _good(task_id="ledger-b"))
        b["done"] = True

    def truncate(fd, length):     # B gets its whole append in before A cuts, unless the lock makes it wait
        if threading.get_ident() == a and "thread" not in b:
            b["thread"] = _in_thread(writer_b)
            b["thread"].join(timeout=0.5)
        return real_truncate(fd, length)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "ftruncate", truncate)
    with pytest.raises(OSError, match="Input/output"):
        HR.append(path, _good(task_id="ledger-a"))
    b["thread"].join(timeout=10)
    monkeypatch.undo()
    assert b.get("done") is True
    assert [r["task_id"] for r in HR.read(path)] == ["ledger-1", "ledger-b"]


WRITER = """
import sys, time
from loopmath import harness_runs as HR
path, name, n, go = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
time.sleep(max(0.0, go - time.time()))
for i in range(n):
    HR.append(path, {"schema": HR.SCHEMA, "task_id": f"{name}-{i}", "ext": {"org.example": {"pad": "x" * 30000}}})
"""


def test_two_processes_and_two_threads_appending_at_once_leave_every_row_whole(tmp_path):
    path, n, go = tmp_path / "runs.jsonl", 100, time.time() + 3.0    # every writer starts at `go`
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(path), f"p{k}", str(n), str(go)], env=env)
             for k in (1, 2)]

    def writer(name):
        time.sleep(max(0.0, go - time.time()))
        for i in range(n):
            HR.append(path, {"schema": HR.SCHEMA, "task_id": f"{name}-{i}", "ext": {"org.example": {"pad": "y" * 30000}}})
    threads = [_in_thread(writer, f"t{k}") for k in (1, 2)]
    for t in threads:
        t.join(timeout=120)
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    got = [r["task_id"] for r in HR.read(path)]
    assert sorted(got) == sorted(f"{w}-{i}" for w in ("p1", "p2", "t1", "t2") for i in range(n))
    for w in ("p1", "p2", "t1", "t2"):    # each writer's own rows stay in its order
        assert [x for x in got if x.startswith(w + "-")] == [f"{w}-{i}" for i in range(n)]
    switches = sum(x.split("-")[0] != y.split("-")[0] for x, y in zip(got, got[1:]))
    assert switches > 3, f"the writers did not overlap ({switches} switches)"


def test_appends_under_locked_never_wait_on_themselves_and_others_wait_for_it(tmp_path):
    path = tmp_path / "runs.jsonl"
    HR.append(path, _good())
    order, other = [], {}

    def gate():
        with HR.locked(path) as log:
            assert [r["task_id"] for r in log.read()] == ["ledger-1"]    # read, check, then append
            log.append(_good(task_id="gate-1"))
            HR.append(path, _good(task_id="gate-2"))     # the plain append in the same thread reuses the lock
            with HR.locked(path) as inner:
                inner.append(_good(task_id="gate-3"))
            other["thread"] = _in_thread(HR.append, path, _good(task_id="other"))
            other["thread"].join(timeout=0.5)
            order.append("other waited" if other["thread"].is_alive() else "other did not wait")
            log.append(_good(task_id="gate-4"))
    t = _in_thread(gate)
    t.join(timeout=20)
    assert not t.is_alive(), "an append under locked waited on its own lock"
    other["thread"].join(timeout=10)
    assert order == ["other waited"]
    assert [r["task_id"] for r in HR.read(path)] == ["ledger-1", "gate-1", "gate-2", "gate-3", "gate-4", "other"]


def test_append_refuses_a_file_whose_last_line_is_torn(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(_good()) + "\n" + '{"schema": "loopmath.harn', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no newline"):
        HR.append(path, _good(task_id="ledger-2"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["auto-research-loop", "plugin-evals", "hand-ab"])
def test_each_worked_example_reads_and_meets_the_schema(name):
    path = EXAMPLES / f"{name}.jsonl"
    rows = HR.read(path)
    assert len(rows) >= 2
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_FILE.read_text(encoding="utf-8")))
    for row in rows:
        assert not list(validator.iter_errors(row)), row
        assert (row.get("task") or {}).get("type") in (*TASK_TYPE_IDS, None), row
        if isinstance(row.get("harness"), dict):
            assert set(row["harness"]["layouts"]) <= {"claude-code", "codex", "harness2"}, row


def test_the_spec_names_every_field():
    text = (ROOT / "spec" / "harness-runs.md").read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    for name in schema["properties"]:
        assert f"`{name}`" in text, name
    for name in ("score", "accepted", "judge", "recovered", "usd_source", "cache_read", "cache_write"):
        assert f"`{name}`" in text, name
