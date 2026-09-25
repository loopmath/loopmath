"""Synthetic fixture: a dozen fake sessions plus a few fake dag attempts, the
hook that makes the suite name what it did not run, and the isolation that
keeps the suite out of the user's store.

Shapes mirror parser-spec.md. No real corpus data appears here.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


# Color proof. Python 3.13+ colors argparse help and tracebacks when the
# environment asks for it (FORCE_COLOR, or a TTY), which breaks every test that
# reads help text. Scrub the color switches once, at import, before any test or
# subprocess runs; children inherit os.environ, so `python -m loopmath ...`
# subprocesses are plain too. The suite then passes whatever the caller's shell
# sets, without PYTHON_COLORS=0 NO_COLOR=1 on the command line.
COLOR_FORCING_VARS = ("FORCE_COLOR", "CLICOLOR_FORCE")
for _name in COLOR_FORCING_VARS:
    os.environ.pop(_name, None)
os.environ["NO_COLOR"] = "1"
os.environ["PYTHON_COLORS"] = "0"


# Tests that need parts of the private development tree. The public export
# (scripts/export-public.sh) leaves these paths out, so there
# the tests skip, and the not-run line below names them.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_PRIVATE_PATH = {
    "tests/test_duplicate_guards.py::test_layout_spike_": "layout-spike",
    "tests/test_ocp_conformance.py::test_layout_spike_": "layout-spike",
    "tests/test_graph_labeler.py::test_circularity_script_": "tools/check_label_circularity.py",
    "tests/test_graph_labeler.py::test_grid_": "tools/grid.sh",
    "tests/priors/test_priors_benchmarks.py::test_shipped_file_is_what_the_generator_writes":
        "design/0.1/data/benchmarks/make_benchmarks.py",
    "tests/test_preview_script.py::": "scripts/preview.sh",
}


# Isolation. Before any test module is imported, LOOPMATH_HOME and
# LOOPMATH_CACHE_DIR point at one temp folder for the whole session, whatever
# the caller's shell set, so no test and no `python -m loopmath` subprocess
# (children inherit os.environ) reads or writes ~/.loopmath. A test that needs
# its own folder sets the variables itself (monkeypatch), which wins for that
# test. The guard test (tests/test_home_guard.py) runs last and fails if any
# path under ~/.loopmath appeared, changed or went away during the run; it
# compares listings with mtimes and sizes, and never writes or deletes there.
ISOLATED_VARS = ("LOOPMATH_HOME", "LOOPMATH_CACHE_DIR")
USER_STORE = Path.home() / ".loopmath"
HOME_GUARD_TEST = "tests/test_home_guard.py::"
_USER_STORE_BEFORE = pytest.StashKey[dict]()
_ISOLATION = pytest.StashKey[tuple]()


def user_store_listing(root: Path = USER_STORE) -> dict[str, tuple[int, int]]:
    """Every path under `root`, `root` included, with its mtime (ns) and size.

    Empty when `root` does not exist. Symlinks are listed, not followed."""
    listing: dict[str, tuple[int, int]] = {}
    try:
        st = os.lstat(root)
    except OSError:
        return listing
    listing[str(root)] = (st.st_mtime_ns, st.st_size)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            listing[path] = (st.st_mtime_ns, st.st_size)
    return listing


def pytest_configure(config):
    config.stash[_USER_STORE_BEFORE] = user_store_listing()
    folder = tempfile.mkdtemp(prefix="loopmath-tests-")
    saved = {name: os.environ.get(name) for name in ISOLATED_VARS}
    config.stash[_ISOLATION] = (folder, saved)
    os.environ["LOOPMATH_HOME"] = os.path.join(folder, "home")
    os.environ["LOOPMATH_CACHE_DIR"] = os.path.join(folder, "cache")


def pytest_unconfigure(config):
    folder, saved = config.stash.get(_ISOLATION, (None, {}))
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if folder:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def session_store_folder(pytestconfig) -> Path:
    """The temp folder this session's LOOPMATH_HOME and LOOPMATH_CACHE_DIR sit in."""
    return Path(pytestconfig.stash[_ISOLATION][0])


@pytest.fixture
def store_listing():
    """`user_store_listing`, for tests of the guard itself."""
    return user_store_listing


@pytest.fixture
def user_store_changes(pytestconfig):
    """A function giving the paths under ~/.loopmath (or `root`) that were added,
    changed or removed since the session started, as three sorted lists."""
    def changes(root: Path = USER_STORE, before: dict | None = None):
        before = pytestconfig.stash[_USER_STORE_BEFORE] if before is None else before
        after = user_store_listing(root)
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
        return added, changed, removed
    return changes


def pytest_collection_modifyitems(config, items):
    for item in items:
        for prefix, need in NEEDS_PRIVATE_PATH.items():
            if item.nodeid.startswith(prefix) and not os.path.exists(os.path.join(_ROOT, need)):
                item.add_marker(pytest.mark.skip(reason=f"needs {need}, which the public tree leaves out"))
    # The guard compares ~/.loopmath with its listing from before the run, so it runs last.
    items.sort(key=lambda item: item.nodeid.startswith(HOME_GUARD_TEST))


# pytest's own last line counts what it did not run ("1 deselected") without
# naming it, so a reader cannot tell which check was absent: the packaging
# smoke test is deselected by the `-m "not packaging"` in pyproject.toml on
# every ordinary run, and a skip can appear from a missing tool. These node
# ids print on one further line, after that count, so the last thing the suite
# says is which tests it did not run.
_NOT_RUN_LABELS = ("deselected", "skipped")


def _node_ids(reports) -> list[str]:
    """Node ids of `reports`, in report order, without repeats.

    `stats["skipped"]` holds one report per skipped test plus a collect report
    for a module-level skip; an xfailing test ran, so it is not listed here.
    """
    seen: list[str] = []
    for report in reports:
        if getattr(report, "wasxfail", None) is not None:
            continue
        node_id = getattr(report, "nodeid", None)
        if node_id and node_id not in seen:
            seen.append(node_id)
    return seen


def not_run_line(stats: dict) -> str:
    """One line naming every collected test the run did not execute, or ""."""
    groups = []
    for label in _NOT_RUN_LABELS:
        node_ids = _node_ids(stats.get(label, ()))
        if node_ids:
            groups.append(f"{len(node_ids)} {label}: {', '.join(node_ids)}")
    return "; ".join(groups)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Print the not-run line after the terminal reporter's own final line.

    Called first among the session-finish wrappers, so this wrapper is the
    outermost one and the code after the yield runs last of all.
    """
    result = yield
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        line = not_run_line(reporter.stats)
        if line:
            reporter.write_line(line)
    return result


def _session(
    sid,
    model,
    effort_sig,
    output_tokens,
    proxy,
    project="/Users/x/Workspace/proj-a",
    error_events=0,
    n_turns_user=3,
    subagent=False,
):
    return {
        "session_id": sid,
        "tool": "claude-code",
        "source_path": f"/fake/{sid}.jsonl",
        "project": project,
        "project_class": "other-dev",
        "started_at": "2026-08-20T10:00:00+00:00",
        "ended_at": "2026-08-20T10:30:00+00:00",
        "duration_s": 1800.0,
        "models_used": [model] if model else [],
        "primary_model": model,
        "reasoning_effort_signals": effort_sig,
        "n_turns_user": n_turns_user,
        "n_turns_assistant": 5,
        "n_tool_calls": 4,
        "tool_call_breakdown": {"Bash": 4},
        "n_subagents_spawned": 0,
        "parallelism_signals": ["subagent_transcript"] if subagent else [],
        "mcp_servers_attached": [],
        "tokens": {"input": 100, "output": output_tokens, "cache_read": 0, "cache_write": 0},
        "outcome": {
            "tests_run": False,
            "tests_passed_signal": None,
            "user_interrupts": 0,
            "error_events": error_events,
            "ended_by": "completed" if proxy == "accepted" else "unknown",
            "acceptance_proxy": proxy,
        },
        "window_flag": "fully_in_window",
        "extras": {"parse_warnings": 0},
    }


@pytest.fixture
def fake_sessions():
    cheap = [
        _session(f"cheap-{i}", "claude-opus-5", ["assistant.effort=medium(10)"], 1000 + i, "accepted")
        for i in range(4)
    ]
    costly = [
        _session(
            f"costly-{i}",
            "claude-fable-5",
            ["assistant.effort=xhigh(20)", "thinking_blocks(5)"],
            50000 + i,
            "accepted" if i < 2 else "unknown",
        )
        for i in range(4)
    ]
    stubs = [
        _session(f"stub-{i}", None, [], 0, "unknown", error_events=1, n_turns_user=0)
        for i in range(3)
    ]
    sub = [_session("sub-1", "claude-opus-5", [], 500, "unknown", subagent=True)]
    return cheap + costly + stubs + sub  # 12 sessions


@pytest.fixture
def fake_attempts():
    def att(task, n, model, result="done", evidence="verified", cause="initial"):
        return {
            "run_id": "run-fake-01",
            "task_id": task,
            "task_kind": "impl",
            "attempt_id": f"{task}·a{n}",
            "n": n,
            "actor": "x",
            "model": model,
            "state": "done",
            "started_at": "2026-08-20T10:00:00Z",
            "ended_at": "2026-08-20T10:10:00Z",
            "locator_pane": None,
            "cause": cause,
            "outcome_result": result,
            "outcome_evidence": evidence,
        }

    return [
        att("T1", 1, "opus5·medium"),
        att("T2", 1, "opus5·medium"),
        att("T3", 1, "opus5·medium", result="rejected", evidence="verified"),
        att("T3", 2, "opus5·medium", cause="sent_back"),
        att("T4", 1, "fable5·xhigh"),
        att("T5", 1, "claude-fable-5·xhigh"),
        att("T6", 1, "fable·xhigh", result="failed"),
        att("T7", 1, "opus5.xhigh"),  # typo form
        att("T8", 1, None),  # null label
        att("T9", 1, "-"),  # dash label
    ]


@pytest.fixture
def fake_runs():
    return [
        {
            "run_id": "run-fake-01",
            "contract_version": 3,
            "dag_depth": 5,
            "max_parallel_width": 2,
            "n_tasks": 9,
            "n_attempts_total": 10,
        }
    ]


@pytest.fixture
def corpus_dir(tmp_path, fake_sessions, fake_attempts, fake_runs):
    def dump(name, records):
        (tmp_path / name).write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )

    dump("sessions.jsonl", fake_sessions)
    dump("dag-attempts.jsonl", fake_attempts)
    dump("dag-runs.jsonl", fake_runs)
    dump("session-dag-join.jsonl", [])
    return tmp_path
