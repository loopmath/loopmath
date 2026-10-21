"""Optional `stages` and `workflow` on a harness-runs row: a multi-stage rollout becomes one OCP run with one piece
per stage role and one attempt per stage, so `posterior --level role` and `check-change from-runs` see the roles.
A row without them gives the same document as before (spec/harness-runs.md section 9)."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re

import pytest

from loopmath import harness_runs as HR
from loopmath.harness_runs.ocp import NS, run_id_for
from loopmath.harness_runs.rows import check_row
from loopmath.ocp.emit import validate_strict
from loopmath.workflows.format import ID_RE, catalog

WHEN = "2026-10-18T17:14:02-07:00"


def _row(**over) -> dict:
    row = {"schema": HR.SCHEMA, "task_id": "acme-tool.1234567", "arm": "plan-review", "agent": "codex",
           "model": "gpt-6-sol", "effort": "high", "pair": "acme-tool.1234567#77", "experiment": "stage-test",
           "started_at": "2026-10-18T17:10:00-07:00", "ended_at": WHEN,
           "outcome": {"name": "pass_rate", "score": 0.62, "scale": "fraction"}}
    row.update(over)
    return row


def _stages() -> list[dict]:
    return [
        {"role": "planner", "agent": "codex", "model": "gpt-6-astra", "effort": "xhigh",
         "cost": {"tokens": {"input": 1000, "cache_read": 800, "output": 200}}},
        {"role": "developer", "agent": "codex", "model": "gpt-6-sol", "effort": "high",
         "cost": {"tokens": {"input": 5000, "cache_read": 4000, "output": 900}}},
        {"role": "reviewer", "agent": "codex", "model": "gpt-6-luna", "effort": "low", "round": 1,
         "cost": {"usd": 0.01, "usd_source": "test"}},
        {"role": "developer", "piece": "implement", "agent": "codex", "model": "gpt-6-sol", "effort": "high",
         "round": 2, "cost": {"usd": 0.05, "usd_source": "test"}},
        {"role": "reviewer", "agent": "codex", "model": "gpt-6-luna", "effort": "low", "round": 2,
         "cost": {"usd": 0.02, "usd_source": "test"}},
    ]


def _strict_errors(doc: dict) -> list:
    return [f for f in validate_strict(doc) if f["level"] == "error"]


def test_row_without_stages_is_unchanged():
    row = _row(cost={"usd": 0.42})
    doc = HR.to_ocp(copy.deepcopy(row))
    assert doc["run"]["configuration"]["workflow"]["id"] == "solo"
    assert len(doc["attempts"]) == 1
    assert doc["attempts"][0]["cost"]["usd"] == 0.42


def test_check_row_accepts_stages_and_workflow():
    assert check_row(_row(stages=_stages(), workflow="plan-dev-review")) == []


@pytest.mark.parametrize("bad, field", [
    ([], "stages"),
    ("plan", "stages"),
    ([{"model": "gpt-6-sol"}], "stages.0.role"),
    ([{"role": ""}], "stages.0.role"),
    ([{"role": "planner", "round": 0}], "stages.0.round"),
    ([{"role": "planner", "status": "running"}], "stages.0.status"),
    ([{"role": "planner", "cost": {"usd": -1}}], "stages.0.cost.usd"),
    ([{"role": "planner", "piece": "a b"}], "stages.0.piece"),
    ([{"role": "lead dev"}], "stages.0.role"),
    ([{"role": "planner", "started_at": "yesterday"}], "stages.0.started_at"),
])
def test_check_row_names_the_bad_stage_field(bad, field):
    problems = check_row(_row(stages=bad))
    assert problems and any(p.startswith(field) for p in problems), problems


@pytest.mark.parametrize("over", [
    {"stages": []}, {"stages": "plan"}, {"stages": [{"model": "gpt-6-sol"}]}, {"stages": [{"role": ""}]},
    {"stages": [{"role": "planner", "round": 0}]}, {"stages": [{"role": "planner", "status": "running"}]},
    {"stages": [{"role": "planner", "cost": {"usd": -1}}]}, {"stages": [{"role": "planner", "piece": "a b"}]},
    {"stages": [{"role": "planner", "started_at": "yesterday"}]}, {"stages": _stages(), "cost": {"usd": 1.0}},
    {"stages": _stages(), "workflow": ""}, {"workflow": "dev-only"},
])
def test_the_published_schema_refuses_the_same_stage_rows(over):
    import json
    from pathlib import Path

    import jsonschema

    path = Path(__file__).resolve().parents[2] / "spec" / "harness-runs-0.1.schema.json"
    validator = jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))
    assert not list(validator.iter_errors(_row(stages=_stages(), workflow="plan-dev-review")))
    assert check_row(_row(**over)) and list(validator.iter_errors(_row(**over)))


def test_top_level_cost_with_stages_is_refused():
    problems = check_row(_row(stages=_stages(), cost={"usd": 1.0}))
    assert any(p.startswith("cost") and "stages" in p for p in problems), problems


def test_one_piece_runs_one_role():
    problems = check_row(_row(stages=[{"role": "developer", "piece": "work"}, {"role": "reviewer", "piece": "work"}]))
    assert any(p.startswith("stages.1.role") and "'work'" in p for p in problems), problems
    # two words for one role are one role
    assert check_row(_row(stages=[{"role": "developer"}, {"role": "implementer", "round": 2}])) == []


def test_workflow_must_be_a_name():
    assert any(p.startswith("workflow") for p in check_row(_row(stages=_stages(), workflow="")))
    assert any(p.startswith("workflow") for p in check_row(_row(workflow="dev-only")))  # needs stages


def test_stages_become_pieces_and_attempts():
    doc = HR.to_ocp(_row(stages=_stages(), workflow="plan-dev-review"))
    assert _strict_errors(doc) == []
    wf = doc["run"]["configuration"]["workflow"]
    assert wf["id"] == "plan-dev-review"
    assert [(p["id"], p["role"]) for p in wf["pieces"]] == [("plan", "planner"), ("implement", "implementer"),
                                                           ("review", "reviewer")]
    atts = doc["attempts"]
    assert [(a["node"], a["round"], a["model"]["id"]) for a in atts] == [
        ("plan", 1, "gpt-6-astra"), ("implement", 1, "gpt-6-sol"), ("review", 1, "gpt-6-luna"),
        ("implement", 2, "gpt-6-sol"), ("review", 2, "gpt-6-luna")]
    assert [a["effort"] for a in atts] == ["xhigh", "high", "low", "high", "low"]
    # each attempt carries its own cost: tokens priced from loopmath's table, dollars as given
    assert atts[0]["cost"]["usd"] > 0 and atts[1]["cost"]["usd"] > atts[0]["cost"]["usd"] / 100
    assert [a["cost"]["usd"] for a in atts[2:]] == [0.01, 0.05, 0.02]
    # the outcome is observed at the last stage's attempt
    sig = doc["run"]["signals"][0]
    assert sig["at_attempt"] == atts[-1]["id"]
    assert doc["run"]["labels"]["arm"] == "plan-review"
    assert doc["run"]["ext"]["dev.loopmath.pair"]["pair"] == "acme-tool.1234567#77"


def test_parallel_stages_of_one_piece_widen_it():
    st = [{"role": "developer", "model": "gpt-6-sol", "effort": "high", "cost": {"usd": 1.0}},
          {"role": "reviewer", "model": "gpt-6-luna", "effort": "low", "cost": {"usd": 0.1}},
          {"role": "reviewer", "model": "gpt-6-astra", "effort": "max", "cost": {"usd": 0.2}}]
    doc = HR.to_ocp(_row(stages=st, workflow="two-reviewers"))
    assert _strict_errors(doc) == []
    widths = {p["id"]: p.get("width", 1) for p in doc["run"]["configuration"]["workflow"]["pieces"]}
    assert widths == {"implement": 1, "review": 2}
    assert [a["model"]["id"] for a in doc["attempts"] if a["node"] == "review"] == ["gpt-6-luna", "gpt-6-astra"]


def test_stages_outside_the_catalog_make_a_chain():
    st = [{"role": "planner", "piece": "plan"}, {"role": "planner", "piece": "split"},
          {"role": "developer", "cost": {"usd": 1.0}}, {"role": "developer", "cost": {"usd": 1.0}},
          {"role": "reviewer", "cost": {"usd": 0.1}}]
    doc = HR.to_ocp(_row(stages=st))
    assert _strict_errors(doc) == []
    wf = doc["run"]["configuration"]["workflow"]
    assert wf["id"] == "stages-plan-split-implement-review"
    assert [(p["id"], p["width"]) for p in wf["pieces"]] == [("plan", 1), ("split", 1), ("implement", 2),
                                                             ("review", 1)]
    assert wf["control"]["gates"] == ["review"] and wf["control"]["repair"] == {"review": "implement"}


def _workflow_ids(doc: dict) -> tuple[set[str], set[str], str]:
    wf = doc["run"]["configuration"]["workflow"]
    return {p["id"] for p in wf["pieces"]}, {a["id"] if isinstance(a, dict) else a for a in wf["artifacts"]}, wf["id"]


# pieces that take the names of the chain's own artifacts and gates
NAMED_LIKE_THE_CHAIN = [
    [{"role": "worker", "piece": "issue"}],
    [{"role": "worker", "piece": "repo"}, {"role": "reviewer", "piece": "issue"}],
    [{"role": "developer", "piece": "a"}, {"role": "reviewer", "piece": "a.out"}],
    [{"role": "developer", "piece": "issue.out"}, {"role": "reviewer", "piece": "issue"}],
    [{"role": "developer", "piece": "g_review"}, {"role": "reviewer"}],
]
# valid piece ids whose derived names pass 200 characters
LONG = [
    [{"role": "developer", "piece": "d" * 200}, {"role": "reviewer"}],
    [{"role": "developer", "piece": "d" * 120}, {"role": "reviewer", "piece": "r" * 120}],
    [{"role": "developer", "piece": "d" * 199}, {"role": "reviewer", "piece": "r" * 200}],
]


@pytest.mark.parametrize("st", NAMED_LIKE_THE_CHAIN + LONG)
def test_any_valid_piece_id_converts(st):
    row = _row(stages=st)
    assert check_row(row) == []
    doc = HR.to_ocp(row)
    assert _strict_errors(doc) == []
    pieces, arts, wid = _workflow_ids(doc)
    assert pieces == {s["piece"] if "piece" in s else "review" for s in st}
    assert not pieces & arts
    assert all(ID_RE.match(i) for i in arts | {wid})
    assert HR.to_ocp(_row(stages=st)) == doc


def _derived_id(st: list[dict]) -> str:
    return _workflow_ids(HR.to_ocp(_row(stages=st)))[2]


def test_derived_workflow_ids_differ_between_lists_of_pieces():
    lists = [
        [{"role": "worker", "piece": "a-b"}],
        [{"role": "worker", "piece": "a"}, {"role": "worker", "piece": "b"}],
        [{"role": "worker", "piece": "a"}, {"role": "worker", "piece": "b-c"}],
        [{"role": "worker", "piece": "a-b"}, {"role": "worker", "piece": "c"}],
        [{"role": "developer", "piece": "d" * 190 + "1"}, {"role": "reviewer"}],
        [{"role": "developer", "piece": "d" * 190 + "2"}, {"role": "reviewer"}],
        [{"role": "developer"}, {"role": "reviewer"}, {"role": "tester"}],
    ]
    ids = [_derived_id(st) for st in lists]
    assert len(set(ids)) == len(ids)
    assert all(len(i) <= 200 and ID_RE.match(i) for i in ids)
    assert ids[-1] == "stages-implement-review-test"
    assert re.fullmatch(r"stages-a-b--[0-9a-f]{12}", ids[0])


def test_a_plain_join_cannot_spell_a_hashed_id():
    hashed = _derived_id([{"role": "worker", "piece": "a-b"}])
    tag = hashed.rsplit("--", 1)[1]
    assert tag == hashlib.sha256(json.dumps(["a-b"]).encode()).hexdigest()[:12]
    plain = [{"role": "worker", "piece": p} for p in ("a", "b", tag)]
    assert _derived_id(plain) == f"stages-a-b-{tag}" != hashed
    long = [{"role": "worker", "piece": "d" * 200}]
    assert "--" in _derived_id(long) and len(_derived_id(long)) <= 200


def test_derived_workflow_id_names_the_pieces():
    assert _derived_id([{"role": "mutator", "piece": "mutate"}]) == "stages-mutate"
    # the id names the pieces, not their roles: give `workflow` to tell such shapes apart
    assert _derived_id([{"role": "developer"}, {"role": "reviewer", "piece": "check"}]) == \
        _derived_id([{"role": "developer"}, {"role": "tester", "piece": "check"}]) == "stages-implement-check"


def test_adapt_converts_pieces_named_like_the_chain(home, tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    for i, st in enumerate(NAMED_LIKE_THE_CHAIN + LONG):
        HR.append(rows, _row(id=f"edge{i}", pair=f"acme-tool.1234567#{i}", stages=st))
    code, out, err = _lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"))
    assert code == 0, err
    docs = [json.loads(f.read_text()) for f in sorted((tmp_path / "ocp").rglob("*.json"))]
    assert len(docs) == len(NAMED_LIKE_THE_CHAIN + LONG)
    assert all(_strict_errors(d) == [] for d in docs)


def test_catalog_workflow_when_roles_match_and_no_workflow_given():
    st = [s for s in _stages() if s.get("round", 1) == 1]
    doc = HR.to_ocp(_row(stages=st))
    wf = doc["run"]["configuration"]["workflow"]
    assert wf["id"] == "plan_implement_review"
    assert catalog()["plan_implement_review"].pieces[0].id == "plan"


def test_stage_cost_unknown_keeps_run_cost_unknown():
    st = _stages()
    st[2]["cost"] = {"complete": False, "tokens": {"input": 10, "output": 5}}
    doc = HR.to_ocp(_row(stages=st))
    assert "cost" not in doc["attempts"][2]
    assert doc["attempts"][2]["ext"]["dev.loopmath.adapter.harness-runs"]["cost"]["complete"] is False


def test_foreign_stage_text_is_kept_and_makes_the_document_full():
    assert HR.to_ocp(_row(stages=_stages()))["privacy"]["profile"] == "metadata_only"
    st = _stages()
    st[0]["ext"] = {"org.example": {"note": "the plan said so"}, "dev.loopmath.mark": {"seen": "yes"}}
    st[1]["transcript"] = "the developer said so"
    doc = HR.to_ocp(_row(stages=st))
    assert doc["privacy"]["profile"] == "full"
    assert doc["run"]["ext"][NS]["retained_text"] == ["stages.0.ext.org.example.note", "stages.1.transcript"]
    first, second = doc["attempts"][0]["ext"][NS], doc["attempts"][1]["ext"][NS]
    assert first["stage_ext"] == st[0]["ext"] and second["extra"] == {"transcript": "the developer said so"}
    assert (first["stage"], first["role"], second["role"]) == (0, "planner", "developer")


def test_every_stage_member_path_is_documented():
    from pathlib import Path

    def paths(value, prefix=""):
        out = set()
        if isinstance(value, dict):
            for key, child in value.items():
                out |= {f"{prefix}{key}"} | paths(child, f"{prefix}{key}.")
        return out

    st = _stages()
    st[0].update(ext={"org.example": {"note": "x"}}, transcript="y")
    st[2]["cost"] = {"complete": False, "incomplete": "cut", "tokens": {"input": 3}}
    doc = HR.to_ocp(_row(stages=st, workflow="plan-dev-review"))
    found = set().union(*(paths(a["ext"][NS]) for a in doc["attempts"])) | paths(doc["run"]["ext"][NS])
    text = (Path(__file__).resolve().parents[2] / "docs" / "adapters.md").read_text(encoding="utf-8")
    missing = [p for p in sorted(found) if not p.startswith(("stage_ext.", "extra.", "row_ext."))
               and f"| `{NS}` | `{p}` |" not in text]
    assert {"stage", "role", "stage_ext", "extra"} <= found and missing == [], missing


def test_same_row_same_document():
    a = HR.to_ocp(_row(stages=_stages(), workflow="plan-dev-review"))
    b = HR.to_ocp(_row(stages=_stages(), workflow="plan-dev-review"))
    assert a == b


# ---------------------------------------------------------------- end to end: adapt, import, compare, posterior
def _lm(capsys, *argv):
    import json

    from loopmath import cli

    capsys.readouterr()
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if "--json" in argv and out.strip() else out), err


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    h.mkdir()
    return h


def _pair_rows(n: int) -> list[dict]:
    """n chains: each gives its unreviewed read (arm unreviewed: the developer's diff) and its final (arm reviewed: the
    reviewed diff), paired on the chain; the reviewer lifts the pass rate by 0.1 on every chain."""
    rows = []
    for i in range(n):
        task, pair = f"acme-tool{i % 4}.1234567", f"acme-tool{i % 4}.1234567#{i}"
        dev = {"role": "developer", "agent": "codex", "model": "gpt-6-sol", "effort": "high", "cost": {"usd": 1.0}}
        rev = {"role": "reviewer", "agent": "codex", "model": "gpt-6-luna", "effort": "low", "cost": {"usd": 0.2}}
        base = 0.3 + 0.05 * (i % 5)
        common = {"schema": HR.SCHEMA, "task_id": task, "pair": pair, "experiment": "stage-test",
                  "task": {"type": "feature", "repo": "acme/tool"}}
        rows.append({**common, "id": f"c{i}:r0", "arm": "unreviewed", "stages": [dev], "workflow": "dev-only",
                     "outcome": {"name": "pass_rate", "score": base, "scale": "fraction"}})
        rows.append({**common, "id": f"c{i}:final", "arm": "reviewed", "stages": [dev, rev], "workflow": "dev-review",
                     "outcome": {"name": "pass_rate", "score": base + 0.1, "scale": "fraction"}})
    return rows


def test_from_runs_pairs_stage_rows_on_the_chain(home, tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    for r in _pair_rows(12):
        HR.append(rows, r)
    assert _lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = _lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = _lm(capsys, "check-change", "from-runs", "--base-label", "arm=unreviewed", "--cand-label", "arm=reviewed",
                         "--rule", "score:pass_rate", "--max-pairs", "12", "--unit", "run",
                         "--json")  # runs, not cases: the pairing on the pair key
    assert code == 0, err
    cmp = out["comparison"]
    assert cmp["n_pairs"] == 12
    assert cmp["gain"]["mean"] == pytest.approx(0.1, abs=1e-9)
    # the per-run costs are the compare's estimates, near the stages' sums
    assert cmp["cost_cand"]["mean"] == pytest.approx(1.2, rel=0.05)
    assert cmp["cost_base"]["mean"] == pytest.approx(1.0, rel=0.05)


def _adversarial_rows() -> tuple[list[dict], set[tuple[str, str]], float]:
    """12 chains on four tasks with distinct gains; the base arm shuffled and started in reverse order; two
    base chains with no final and one final with no base. (rows, the true (base, cand) run pairs, their mean
    gain): pairing on the task in start order, or on row order, gives other pairs and another mean."""
    rows = _pair_rows(12)
    for r in rows:
        n = int(r["pair"].rsplit("#", 1)[1])
        r["started_at"] = f"2026-10-18T{(23 - n) if r['arm'] == 'unreviewed' else n:02d}:00:00-07:00"
        if r["arm"] == "reviewed":
            r["outcome"]["score"] = round(r["outcome"]["score"] - 0.2 + 0.03 * n, 6)
    base = {r["pair"]: r for r in rows if r["arm"] == "unreviewed"}
    cand = {r["pair"]: r for r in rows if r["arm"] == "reviewed"}
    del cand["acme-tool0.1234567#0"], cand["acme-tool1.1234567#5"], base["acme-tool2.1234567#6"]
    keys = sorted(base.keys() & cand.keys())
    gains = [cand[k]["outcome"]["score"] - base[k]["outcome"]["score"] for k in keys]
    assert len({round(g, 9) for g in gains}) == len(keys)
    shuffled = list(base.values())
    random.Random(7).shuffle(shuffled)
    return (shuffled + list(cand.values()), {(run_id_for(base[k]), run_id_for(cand[k])) for k in keys},
            sum(gains) / len(gains))


def test_from_runs_pairs_on_the_pair_key_not_on_the_task(home, tmp_path, capsys):
    rows, truth, mean = _adversarial_rows()
    path = tmp_path / "rows.jsonl"
    for r in rows:
        HR.append(path, r)
    assert _lm(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = _lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = _lm(capsys, "check-change", "from-runs", "--base-label", "arm=unreviewed", "--cand-label", "arm=reviewed",
                         "--rule", "score:pass_rate", "--max-pairs", "24", "--unit", "run",
                         "--json")  # runs, not cases: the pairing on the pair key
    assert code == 0, err
    cmp = out["comparison"]
    assert {(p["base"], p["cand"]) for p in cmp["pairs"]} == truth
    assert cmp["n_pairs"] == len(truth) == 9
    assert cmp["counts"]["lone_runs"] == {"base": 2, "cand": 1}
    assert cmp["gain"]["mean"] == pytest.approx(mean, abs=0.01)



def test_posterior_role_level_sees_the_stage_roles(home, tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    for r in _pair_rows(6):
        HR.append(rows, r)
    assert _lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"), "--accept-at", "0.5")[0] == 0
    code, _, err = _lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, _, err = _lm(capsys, "fit", "--no-prior", "--json")
    assert code == 0, err
    code, out, err = _lm(capsys, "posterior", "--level", "role", "--json")
    assert code == 0, err
    keys = {e["key"] for e in out["levels"]["role"]}
    # public keys carry no level prefix (belief/state.py NodeSummary.key)
    assert {"implementer", "reviewer"} <= keys, keys


# ---------------------------------------------------------------- rows without stages: frozen at 0.3.1
def _no_stage_rows() -> list[dict]:
    """The spec examples and hand rows over each cost, outcome and text branch, none with stages."""
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parents[2] / "spec" / "examples" / "harness-runs"
    rows = [json.loads(line) for name in ("hand-ab.jsonl", "auto-research-loop.jsonl", "plugin-evals.jsonl")
            for line in (here / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows += [
        _row(cost={"usd": 0.42, "usd_source": "reported"}),
        _row(cost={"tokens": {"input": 1200, "cache_read": 300, "cache_write": 40, "output": 500}}),
        _row(cost={"complete": False, "incomplete": "cut", "partial_usd": 0.1, "tokens": {"input": 3}}, status="failed"),
        _row(outcome={"accepted": True, "by": "a judge"}, ext={"org.example": {"note": "kept"}}, note="extra text"),
        _row(model=None, agent=None, effort=None, started_at=None, ended_at=None, arm=None, pair=None,
             task={"type": "bug_fix", "repo": "acme/api", "title": "Fix it", "base_commit": "abc123"},
             tool={"name": "hand", "run_id": "r1", "stage": "final"}),
    ]
    return rows


def _frozen_digest(doc: dict) -> str:
    import hashlib
    import json

    doc = copy.deepcopy(doc)
    doc["producer"]["version"] = "<version>"
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _no_stage_digests() -> list[str]:
    from pathlib import Path

    base = Path(__file__).resolve().parents[2] / "spec" / "examples" / "harness-runs"
    return [_frozen_digest(HR.to_ocp(r, base_dir=base, observed_default=WHEN)) for r in _no_stage_rows()]


# written by _no_stage_digests() under the 0.3.1 release (before stages), from a git archive of its src
NO_STAGE_DIGESTS: list[str] = [
    "565925c8f3b479f2", "4a177ec1dca7cbd7", "b238b771664ef574", "163d6ea74230d086", "9f60a9377d758d69", "272adbdcf62e113c",
    "238d19e62552d303", "1b6f4e205b432c4b", "3f24b36bc8c1202e", "dd63b54557316f11", "268e0e559e81e1c6", "0adb221f9b9d6a10",
    "9ea26a42f844348c", "925825af3db3c91a", "18ef68471346ba03", "fba40799bcb118f1", "e47dc2f650550eed", "baedc9dfd2778205",
    "ba8826d05a9180e1", "24725aea29ca7dab", "b06312a03e18d7ca", "55bd8c3f39c87247",
]


def test_rows_without_stages_give_the_documents_of_0_3_1():
    assert _no_stage_digests() == NO_STAGE_DIGESTS
