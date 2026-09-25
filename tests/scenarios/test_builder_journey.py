"""The builder journey, synthetic: from the planning page's Customize line to `loopmath builder` and its numbers.

The designed runs of `test_designed_import.py` (two tasks of one type and repo, a score target of
heldout_perf >= 2400) are imported and fitted once, with config `models.allowed` naming the two models they ran
(the models this user offers); `recommend --json --html PATH` stores a recommendation and
writes the planning page. Then the command the page's Customize line gives runs as the user runs it,
`python -m loopmath builder --rec REC --start CFG --no-open`, and the page's calls are made against it on
127.0.0.1: `GET /`, `GET /api/context`, `POST /api/predict` and `POST /api/predict_many`. It checks:

- the context answers for the stored recommendation, opens on the given configuration and lists the choices
  `recommend` gave, by option number and key, and names the goal's configuration;
- a choice's configuration predicts to the numbers `recommend` gave it, the goal's chance included;
- every number carries nested bands (chance, run cost, cost per accepted result, chance within the attempts);
  a reply gives the expected review rounds and each gated piece's pass chance;
- predict_many answers each configuration exactly as predict does, in order, and one bad configuration gets an
  error tied to its piece without stopping the others;
- the catalog counts the runs behind each model by role and effort, and the runs of each role;
- the planning page carries the Customize line, and the builder page calls the live API;
- Ctrl+C stops the command with exit 0, and nothing is written to the store.

Nothing here comes from a real store: every document is built by `write_docs`.
"""

from __future__ import annotations

import copy
import http.client
import json
import os
import re
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scenarios.test_designed_import import LUNA, REPO, SOL, TARGET, call, call_json, write_docs

TITLE = "Feature task in example/bench to reach heldout_perf >= 2400"
PLAN = ("recommend", "--type", "feature", "--repo", REPO, "--title", TITLE, "--target", TARGET)
SRC = str(Path(__file__).resolve().parents[2] / "src")
NAMES = ("chance", "run_cost_usd", "cost_per_accepted_usd", "p_accepted_within")
LEVELS = ("50", "80", "90", "95")
UNKNOWN_MODEL = "gpt-0-unknown"


def request(port: int, method: str, path: str, body: object = None) -> tuple[int, object]:
    """(status, the JSON reply or the text of a page)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        r = conn.getresponse()
        raw = r.read().decode("utf-8")
        return r.status, json.loads(raw) if (r.getheader("Content-Type") or "").startswith("application/json") else raw
    finally:
        conn.close()


def listing(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under the store with its size and modification time."""
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def first_line(proc: subprocess.Popen, timeout: float = 180.0) -> str:
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    return proc.stdout.readline() if ready else ""


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    """The whole journey once, in order; each test reads its step."""
    root = tmp_path_factory.mktemp("builder-journey")
    env = {"LOOPMATH_HOME": str(root / "home"), "LOOPMATH_CACHE_DIR": str(root / "cache"),
           "CLAUDE_CONFIG_DIR": str(root / "logs" / "claude"), "CODEX_HOME": str(root / "logs" / "codex")}
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        (root / "docs").mkdir()
        write_docs(root / "docs")
        code, out, err = call("run", "import", str(root / "docs"), "--finish", "--no-fit")
        assert code == 0, err
        code, out, err = call("config", "set", "models.allowed", f"{SOL}, {LUNA}")  # the models this user offers
        assert code == 0, err
        call_json("fit")
        rec = call_json(*PLAN, "--html", str(root / "plan.html"))
    steps = {"rec": rec, "stored": json.loads((root / "home" / "recs" / f"{rec['rec']}.json").read_text()),
             "plan_page": Path(rec["page"]).read_text(encoding="utf-8"), "before": listing(root / "home")}
    goal = rec["goal"]["config"]
    proc = subprocess.Popen([sys.executable, "-m", "loopmath", "builder", "--rec", rec["rec"], "--start", goal,
                             "--no-open"], env={**os.environ, **env, "PYTHONPATH": SRC},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = first_line(proc)
        assert line.startswith("loopmath builder: http://127.0.0.1:"), (line, proc.stderr.read() if proc.poll() else "")
        steps["url"] = line.split("loopmath builder: ", 1)[1].split()[0]
        port = int(steps["url"].split("http://127.0.0.1:", 1)[1].split("/", 1)[0])
        code, page = request(port, "GET", "/")
        scripts = [request(port, "GET", f"/assets/{name}")[1]
                   for name in dict.fromkeys(re.findall(r"assets/([a-z][a-z0-9_-]*\.js)", page))]
        steps["page"] = (code, "\n".join([page, *(s for s in scripts if isinstance(s, str))]))
        steps["context"] = ctx = request(port, "GET", "/api/context")[1]
        choices = [c for c in ctx["choices"] if c.get("configuration")]
        steps["predict"] = {c["key"]: request(port, "POST", "/api/predict", {"config": c["configuration"]})[1]
                            for c in choices}
        gated = next(c["config"] for c in ctx["candidates"] if c["config"]["workflow"]["control"].get("gates"))
        steps["gated"] = request(port, "POST", "/api/predict", {"config": gated})[1]
        bad = copy.deepcopy(ctx["choices"][0]["configuration"])
        steps["bad_piece"] = piece = bad["workflow"]["pieces"][-1]["id"]
        bad["settings"][piece]["model"] = UNKNOWN_MODEL
        steps["invalid"] = request(port, "POST", "/api/predict", {"config": bad})[1]
        steps["sent"] = [c["configuration"] for c in choices] + [bad]
        steps["many"] = request(port, "POST", "/api/predict_many", {"configs": steps["sent"]})
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=60)
        steps["stop"] = (proc.returncode, line + out, err)
    finally:
        if proc.poll() is None:
            print(f"builder journey: stopping pid {proc.pid} (parent {os.getpid()})", file=sys.stderr)
            proc.kill()
            proc.wait()
    steps["after"] = listing(root / "home")
    yield steps


def by_key(choices: list[dict]) -> dict[str, dict]:
    return {c["key"]: c for c in choices}


def as_recommend(numbers: dict) -> dict:
    """A predict reply's numbers as `recommend --json` gives them: no `chance` (a choice has its own) and no
    `p_accepted_within` band (the builder's own, 0.2.2)."""
    out = {k: v for k, v in numbers.items() if k != "chance"}
    if isinstance(out.get("bands"), dict):
        out["bands"] = {k: v for k, v in out["bands"].items() if k != "p_accepted_within"}
    return out


def assert_nested(bands: dict, where: str) -> None:
    for name in NAMES:
        band = bands[name]
        assert [lvl for lvl in LEVELS if lvl in band] == list(LEVELS), (where, name, band)
        for inner, outer in zip(LEVELS, LEVELS[1:]):
            assert outer_contains(band[outer], band[inner]), (where, name, band)


def outer_contains(outer: list, inner: list) -> bool:
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


# ---------------------------------------------------------------- start
def test_the_customize_command_starts_the_builder_on_127_0_0_1(journey):
    assert journey["url"].startswith("http://127.0.0.1:")
    code, text = journey["page"]
    assert code == 200 and "<html" in text.lower()


def test_the_context_is_the_stored_recommendation_opened_on_the_goal(journey):
    ctx, rec = journey["context"], journey["rec"]
    assert ctx["schema"] == "loopmath.builder.context/1"
    assert ctx["rec"] == rec["rec"] and ctx["fit"]["id"] == rec["fit"]["id"]
    assert ctx["start"]["id"] == rec["goal"]["config"]
    assert ctx["rescue"] == rec["rescue"] and ctx["rescue"]["text"]
    assert [(c["option"], c["key"], c["config"]) for c in ctx["choices"]] == \
        [(c["option"], c["key"], c["config"]) for c in rec["choices"]]


def test_the_context_names_the_goal_configuration(journey):
    ctx = journey["context"]
    assert ctx["goal_config_id"] == journey["rec"]["goal"]["config"] == by_key(ctx["choices"])["goal"]["config"]


# ---------------------------------------------------------------- numbers equal recommend
def test_each_choice_predicts_to_the_numbers_recommend_gave_it(journey):
    candidates = {c["config"]["id"]: c for c in journey["stored"]["candidates"]}
    choices = by_key(journey["rec"]["choices"])
    assert "goal" in journey["predict"]
    for key, out in journey["predict"].items():
        if key == "pair":  # two workflows; its numbers are the pair's, not one configuration's
            continue
        want = candidates[choices[key]["config"]]
        assert out["ok"] and out["config_id"] == want["config"]["id"] and out["known"] is True, (key, out)
        assert as_recommend(out["numbers"]) == want["numbers"], key
    goal = journey["predict"]["goal"]["numbers"]
    assert goal["chance"] == choices["goal"]["chance"]
    assert goal["cost_per_accepted_usd"]["mean"] == choices["goal"]["cost_per_accepted_usd"]["mean"]


def test_every_number_has_nested_bands(journey):
    ctx = journey["context"]
    for key, out in journey["predict"].items():
        assert_nested(out["numbers"]["bands"], f"predict {key}")
    assert_nested(ctx["reference"]["numbers"]["bands"], "reference")
    for c in ctx["choices"]:
        assert_nested(c["bands"], f"choice {c['key']}")
    for c in ctx["candidates"]:
        assert_nested(c["numbers"]["bands"], f"candidate {c['config']['id']}")


def test_a_reply_gives_review_rounds_and_each_gated_pieces_pass_chance(journey):
    out = journey["gated"]
    assert out["ok"], out
    rounds = out["rounds"]
    assert 1.0 <= rounds["mean"] <= rounds["max"] and rounds["lo"] <= rounds["mean"] <= rounds["hi"]
    gates = {g["after"] for g in next(c["config"] for c in journey["context"]["candidates"]
                                      if c["config"]["id"] == out["config_id"])["workflow"]["control"]["gates"]}
    for p in out["pieces"]:
        assert set(p["run_cost_usd"]) == {"mean", "lo", "hi"}
        if p["piece"] in gates:
            g = p["gate_pass"]
            assert 0.0 <= g["lo"] <= g["mean"] <= g["hi"] <= 1.0, p
        else:
            assert p["gate_pass"] is None, p


# ---------------------------------------------------------------- predict_many and errors
def test_predict_many_answers_each_configuration_as_predict_does_in_order(journey):
    code, many = journey["many"]
    assert code == 200 and many["ok"] is True and many["errors"] == []
    results = many["results"]
    assert len(results) == len(journey["sent"])
    assert results[:-1] == list(journey["predict"].values())
    assert results[-1] == journey["invalid"] and results[-1]["ok"] is False


def test_an_error_is_tied_to_its_piece(journey):
    out = journey["invalid"]
    assert out["ok"] is False and "numbers" not in out
    assert any(e["piece"] == journey["bad_piece"] and UNKNOWN_MODEL in e["message"] for e in out["errors"]), out


# ---------------------------------------------------------------- the catalog
def test_the_catalog_counts_the_runs_behind_each_model_by_role_and_effort(journey):
    """From the designed runs: sol solo at five efforts and best of 3 at xhigh, a sol planner over three luna
    workers, luna solo at high; each arm on two tasks."""
    cat = journey["context"]["catalog"]
    models = {m["id"]: m["runs_behind"] for m in cat["models"]}
    assert models[SOL] == {"total": 14, "by_role": {"implementer": 12, "planner": 2},
                           "by_effort": {"implementer": {"low": 2, "medium": 2, "high": 2, "xhigh": 4, "max": 2},
                                         "planner": {"max": 2}}}
    assert models[LUNA] == {"total": 4, "by_role": {"implementer": 2, "worker": 2},
                            "by_effort": {"implementer": {"high": 2}, "worker": {"high": 2}}}
    roles = {r["id"]: r["runs"] for r in cat["roles"]}
    assert (roles["implementer"], roles["planner"], roles["worker"], roles["tester"]) == (14, 2, 2, 0)


# ---------------------------------------------------------------- the pages
def test_the_planning_page_carries_the_customize_line(journey):
    assert "loopmath builder --rec" in journey["plan_page"] and "--start" in journey["plan_page"]


def test_the_builder_page_calls_the_live_api(journey):
    _, text = journey["page"]
    assert "/api/context" in text and "/api/predict_many" in text


# ---------------------------------------------------------------- stop
def test_ctrl_c_stops_it_and_nothing_is_written_to_the_store(journey):
    code, out, err = journey["stop"]
    assert code == 0 and "stopped" in out, err
    assert journey["after"] == journey["before"]
