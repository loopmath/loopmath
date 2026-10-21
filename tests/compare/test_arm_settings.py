"""A check's arms by model and effort (`check-change start --base-set/--cand-set`).

A model switch runs one harness with two models. Each arm keeps its `set`; `run record --check --arm` takes it,
refuses a contradicting `--set`, and `status` shows the models each arm ran and any run whose log ran another.
Synthetic stores. The real-log tests write Claude Code shaped logs into a temp CLAUDE_CONFIG_DIR, with the row
shape of tests/onboard/onboard_fixture.py:56-75. Nothing here is a real session.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from loopmath.price import load_prices
from loopmath.store.home import Store
from tests.compare.test_check_change_cli import SOLO, home, lm, tasks_file  # noqa: F401

OPUS = "implement=claude-code:claude-opus-5-5:high"
SONNET = "implement=claude-code:claude-sonnet-5:high"
REC = ["--workflow", "solo", "--source", "designed"]  # no --set: the arm gives it
STATUS_KEYS_031 = {"check", "base", "cand", "rule", "pairs", "recorded", "open", "preferences", "comparison",
                   "next", "out_of_order"}


def _start(capsys, tmp_path, *extra, n=4):
    return lm(capsys, "check-change", "start", "--base", "opus-5-5-high", "--cand", "sonnet-5-high",
              "--base-set", OPUS, "--cand-set", SONNET, "--tasks", str(tasks_file(tmp_path, n)), *extra, "--json")


def _rec(capsys, home, chk, arm, *extra, verdict="pass"):
    return lm(capsys, "run", "record", "--check", chk, "--arm", arm, *REC,
              "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--verified", f"tests={verdict}",
              "--no-commits", "--no-fit", *extra, "--json")


def _check_file(home: Path, chk: str) -> dict:
    return json.loads((home / "checks" / f"{chk}.json").read_text())


def _snapshot(home: Path) -> dict[str, bytes]:
    """Every file under the store, to show a refusal wrote nothing. `lock` is the store's empty lock file, which
    a run start into a check takes before its first check."""
    return {str(p.relative_to(home)): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _index(home: Path) -> list[dict]:
    path = home / "runs" / "index.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []


# ---------------------------------------------------------------- start


def test_start_keeps_each_arms_set_in_the_check_and_its_json(home, tmp_path, capsys):
    code, out, err = _start(capsys, tmp_path)
    assert code == 0, err
    assert out["base"]["set"] == {"implement": "claude-code:claude-opus-5-5:high"}
    assert out["cand"]["set"] == {"implement": "claude-code:claude-sonnet-5:high"}
    saved = _check_file(home, out["check"])
    assert saved["base"]["set"] == out["base"]["set"] and saved["cand"]["set"] == out["cand"]["set"]


def test_arms_are_still_told_apart_by_label(home, tmp_path, capsys):
    tasks = str(tasks_file(tmp_path, 2))
    code, _, err = lm(capsys, "check-change", "start", "--base", "m", "--cand", "m", "--base-set", OPUS,
                      "--cand-set", SONNET, "--tasks", tasks)
    assert code == 1 and "the same arm" in err, err  # two sets do not make one label two arms
    code, out, err = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--base-set", OPUS,
                        "--cand-set", OPUS, "--tasks", tasks, "--json")
    assert code == 0, err  # two labels on one model: arms that differ by something else
    code, out, err = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--cand-set", SONNET,
                        "--tasks", tasks, "--json")
    assert code == 0, err
    assert "set" not in out["base"] and out["cand"]["set"] == {"implement": "claude-code:claude-sonnet-5:high"}


def test_a_malformed_arm_set_is_refused_with_the_set_form(home, tmp_path, capsys):
    code, _, err = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--base-set", "implement=opus",
                      "--tasks", str(tasks_file(tmp_path, 2)))
    assert code == 1 and "--base-set" in err and "PIECE=HARNESS:MODEL:EFFORT" in err, err
    assert not (home / "checks").exists()


# ---------------------------------------------------------------- run record into an arm with a set


def test_a_run_with_no_set_takes_its_arms_set(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    code, rec, err = _rec(capsys, home, chk, "cand")
    assert code == 0, err
    implement = Store(home).run_doc(rec["run"])["run"]["configuration"]["settings"]["implement"]
    assert implement["model"]["raw"] == "claude-sonnet-5" and implement["effort"] == "high", implement


@pytest.mark.parametrize("given, said", [
    (OPUS, "claude-opus-5-5"),  # another model
    ("implement=codex:claude-sonnet-5:high", "codex"),  # another harness
    ("implement=claude-code:claude-sonnet-5:low", ":low"),  # another effort
    ("implement=claude-code:claude-sonnet-5", ":default"),  # no effort is `default`, not the arm's high
])
def test_a_contradicting_set_is_refused_before_anything_is_written(home, tmp_path, capsys, given, said):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    before = _snapshot(home)
    code, _, err = _rec(capsys, home, chk, "cand", "--set", given)
    assert code == 1 and said in err and "claude-code:claude-sonnet-5:high" in err, err
    assert _snapshot(home) == before
    code, _, err = lm(capsys, "run", "start", "--check", chk, "--arm", "cand", *REC, "--set", given)
    assert code == 1 and said in err, err
    assert _snapshot(home) == before


def test_a_config_rec_or_choice_is_refused_for_an_arm_with_a_set(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    before = _snapshot(home)
    wt = f"implement={home.parent / 'wt'}"
    tail = ["--cwd", wt, "--since", "1h", "--verified", "tests=pass", "--no-commits", "--no-fit"]
    for argv, said in ((["--config", "cfg_000000000000", "--source", "designed"], "--config"),
                       (["--rec", "rec_000000000000", "--workflow", "solo", "--source", "designed"], "--rec"),
                       (["--rec", "rec_000000000000", "--choice", "usual"], "--choice")):
        code, _, err = lm(capsys, "run", "record", "--check", chk, "--arm", "cand", *argv, *tail)
        assert code == 1 and said in err, (argv, err)
        code, _, err = lm(capsys, "run", "start", "--check", chk, "--arm", "cand", *argv)
        assert code == 1 and said in err, (argv, err)
    assert _snapshot(home) == before
    assert _index(home) == [] and _check_file(home, chk)["pairs"][0]["runs"] == {"base": None, "cand": None}


def test_the_arms_own_set_said_again_or_spelled_another_way_is_the_arms(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    configs = []
    for arm, given in (("cand", SONNET), ("cand", "implement=claude-code:sonnet-5:HIGH")):
        code, rec, err = _rec(capsys, home, chk, arm, "--set", given)
        assert code == 0, err
        configs.append(Store(home).run_doc(rec["run"])["run"]["configuration"])
    assert configs[0]["id"] == configs[1]["id"]  # one configuration: the arm's spelling, not the flag's
    assert configs[1]["settings"]["implement"]["model"]["raw"] == "claude-sonnet-5"


def test_run_start_into_an_arm_takes_its_set(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    code, started, err = lm(capsys, "run", "start", "--check", chk, "--arm", "base", *REC, "--json")
    assert code == 0, err
    implement = Store(home).run_doc(started["run"])["run"]["configuration"]["settings"]["implement"]
    assert implement["model"]["raw"] == "claude-opus-5-5" and implement["effort"] == "high", implement
    nxt = next(c for c in out["next"]["commands"] if "--arm base" in c)
    assert "--set implement=claude-code:claude-opus-5-5:high" in nxt


# ---------------------------------------------------------------- status and the next pair


def test_status_shows_each_arms_set_and_the_models_it_ran(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    for arm in ("base", "cand"):
        code, _, err = _rec(capsys, home, chk, arm)
        assert code == 0, err
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    assert st["base"]["set"] == {"implement": "claude-code:claude-opus-5-5:high"}
    assert st["cand"]["set"] == {"implement": "claude-code:claude-sonnet-5:high"}
    assert st["models"] == {"base": ["claude-opus-5-5"], "cand": ["claude-sonnet-5"]}
    assert st["model_mismatch"] == []
    code, text, _ = lm(capsys, "check-change", "status", chk)
    head = text.splitlines()[0]
    assert "claude-opus-5-5" in head and "claude-sonnet-5" in head, head


def test_the_next_pair_lines_carry_each_arms_set(home, tmp_path, capsys):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    lines = out["next"]["commands"]
    base = next(c for c in lines if f"--check {chk} --arm base" in c)
    cand = next(c for c in lines if f"--check {chk} --arm cand" in c)
    assert f"--set {OPUS}" in base and f"--set {SONNET}" in cand, lines


def test_a_check_without_sets_keeps_the_031_shapes(home, tmp_path, capsys):
    """Characterization: passes on 0.3.1 and must keep passing."""
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--json")
    assert code == 0, err
    chk = out["check"]
    assert set(out["base"]) == {"label"} and set(out["cand"]) == {"label"}
    saved = _check_file(home, chk)
    assert set(saved["base"]) == {"label"} and set(saved["cand"]) == {"label"}
    assert not any("--set" in c for c in out["next"]["commands"])
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    assert set(st) - {"schema"} == STATUS_KEYS_031, sorted(st)


# ---------------------------------------------------------------- real logs: what each arm ran, and what it cost


def _claude_log(logs: Path, sid: str, cwd: Path, model: str, *, usage=(1000, 0, 5000, 300), ago_min: float = 10,
                effort: str | None = None):
    """One Claude Code session: a prompt and one assistant turn. Rows as tests/onboard/onboard_fixture.py:56-75;
    usage is (input, cache creation, cache read, output). `effort` is the row field the reader counts
    (src/loopmath/ingest/claude_code.py:280-282)."""
    t0 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=ago_min)

    def ts(t):
        return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    inp, cw, cr, out = usage
    rows = [{"type": "user", "timestamp": ts(t0), "sessionId": sid, "cwd": str(cwd),
             "message": {"role": "user", "content": "Fix the failing test"}},
            {"type": "assistant", "timestamp": ts(t0 + dt.timedelta(minutes=2)), "sessionId": sid, "cwd": str(cwd),
             "requestId": f"req-{sid}",
             "message": {"model": model, "role": "assistant", "content": [{"type": "text", "text": "Fixed."}],
                         "stop_reason": "end_turn",
                         "usage": {"input_tokens": inp, "cache_creation_input_tokens": cw,
                                   "cache_read_input_tokens": cr, "output_tokens": out}}}]
    if effort is not None:
        rows[1]["effort"] = effort
    path = logs / "projects" / str(cwd).replace("/", "-") / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def logged(tmp_path, monkeypatch):
    """A store whose runs settle from the Claude Code logs under tmp_path/logs, not from a fake."""
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "logs"))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    (tmp_path / "wt").mkdir()
    h.mkdir()
    return h


def _rec_session(capsys, chk, arm, sid, *extra):
    return lm(capsys, "run", "record", "--check", chk, "--arm", arm, "--workflow", "solo", "--source", "designed",
              "--session", sid, "--verified", "tests=pass", "--no-commits", "--no-fit", *extra, "--json")


@pytest.mark.parametrize("cand_log_model, flagged", [("claude-opus-5-5", True), ("sonnet-5", False)])
def test_a_run_whose_log_ran_another_model_than_its_arm_is_named(logged, tmp_path, capsys, cand_log_model, flagged):
    _, out, _ = _start(capsys, tmp_path)
    chk = out["check"]
    _claude_log(tmp_path / "logs", "s-base", tmp_path / "wt", "claude-opus-5-5")
    _claude_log(tmp_path / "logs", "s-cand", tmp_path / "wt", cand_log_model, ago_min=8)
    for arm, sid in (("base", "s-base"), ("cand", "s-cand")):
        code, _, err = _rec_session(capsys, chk, arm, sid)
        assert code == 0, err
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    code, text, _ = lm(capsys, "check-change", "status", chk)
    if flagged:
        assert st["model_mismatch"] == [{"pair": "p01", "arm": "cand", "piece": "implement",
                                         "declared": "claude-code:claude-sonnet-5:high",
                                         "ran": "claude-code:claude-opus-5-5:high"}]
        assert st["models"]["cand"] == ["claude-opus-5-5"]
        assert any("p01" in ln and "claude-opus-5-5" in ln and "cand" in ln for ln in text.splitlines()), text
    else:  # `sonnet-5` is Claude Code's spelling of `claude-sonnet-5`, not another model
        assert st["model_mismatch"] == []
        assert st["models"] == {"base": ["claude-opus-5-5"], "cand": ["claude-sonnet-5"]}


def test_cost_splits_by_the_model_each_arm_ran(logged, tmp_path, capsys):
    """Characterization: passes on 0.3.1 with an explicit --set per run. Equal tokens, two models: each run is
    priced at its own model's row in prices.toml, so the dollar ratio is the price ratio."""
    code, out, err = lm(capsys, "check-change", "start", "--base", "opus-5-5-high", "--cand", "sonnet-5-high",
                        "--tasks", str(tasks_file(tmp_path, 4)), "--json")
    assert code == 0, err
    chk = out["check"]
    usage = (1000, 0, 5000, 300)
    _claude_log(tmp_path / "logs", "s-base", tmp_path / "wt", "claude-opus-5-5", usage=usage)
    _claude_log(tmp_path / "logs", "s-cand", tmp_path / "wt", "claude-sonnet-5", usage=usage, ago_min=8)
    runs = {}
    for arm, sid, s in (("base", "s-base", OPUS), ("cand", "s-cand", SONNET)):
        code, rec, err = _rec_session(capsys, chk, arm, sid, "--set", s)
        assert code == 0, err
        runs[arm] = rec["run"]
    st = Store(logged)
    usd = {arm: sum(a["cost"]["usd"] for a in st.run_doc(run)["attempts"]) for arm, run in runs.items()}
    table = load_prices()

    def price(model):
        r = table.rate(model)
        return (usage[0] * r["input"] + usage[2] * r["cache_read"] + usage[3] * r["output"]) / 1e6

    assert usd["base"] == pytest.approx(price("claude-opus-5-5"))
    assert usd["cand"] == pytest.approx(price("claude-sonnet-5"))
    assert usd["cand"] < usd["base"]


@pytest.mark.parametrize("cand_set, logged_effort, flagged", [
    (SONNET, "low", True),  # the arm says high, the log says low
    (SONNET, "high", False),
    (SONNET, None, False),  # the log gives no effort: nothing to hold against the arm
    ("implement=claude-code:claude-sonnet-5:default", "high", False),  # the arm names none: any effort is its
])
def test_effort_counts_when_the_arm_names_one_and_the_log_gives_one(logged, tmp_path, capsys, cand_set,
                                                                    logged_effort, flagged):
    code, out, err = lm(capsys, "check-change", "start", "--base", "opus-5-5-high", "--cand", "sonnet-5",
                        "--base-set", OPUS, "--cand-set", cand_set, "--tasks", str(tasks_file(tmp_path, 4)), "--json")
    assert code == 0, err
    chk = out["check"]
    _claude_log(tmp_path / "logs", "s-cand", tmp_path / "wt", "claude-sonnet-5", effort=logged_effort)
    code, _, err = _rec_session(capsys, chk, "cand", "s-cand")
    assert code == 0, err
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    if flagged:
        assert st["model_mismatch"] == [{"pair": "p01", "arm": "cand", "piece": "implement",
                                         "declared": "claude-code:claude-sonnet-5:high",
                                         "ran": "claude-code:claude-sonnet-5:low"}]
    else:
        assert st["model_mismatch"] == []
