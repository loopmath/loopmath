"""`loopmath builder` 0.2.2 additions (lane 22W, spec 02 `builder`): `POST /api/predict_many`, the four bands
with `p_accepted_within`, `pieces[].gate_pass` and `rounds`, errors tied to pieces, the runs behind each model
and role, `goal_config_id`, and `--start` from any option or candidate with `--rec`. The fit is the search
tests' synthetic one; the only network is 127.0.0.1."""

from __future__ import annotations

import copy
import json

import pytest

from test_builder_api import (  # noqa: F401 - `home` is the shared fixture
    CATALOG, H, TASK, as_recommend, build_session, home, parse, recommend_json, request, session_for, write_runs,
)

from loopmath.builder import predict as P
from loopmath.builder import server as srv_mod
from loopmath.builder.context import context_payload
from loopmath.recommend import engine
from loopmath.recommend.curve import accepted_within

LEVELS = ("50", "80", "90", "95")
BANDS = ("chance", "run_cost_usd", "cost_per_accepted_usd", "p_accepted_within")


def edited(cfg: dict, width: int = 2) -> dict:
    d = copy.deepcopy(cfg)
    d["workflow"]["pieces"][0]["width"] = width
    d["id"] = None
    return d


# ---------------------------------------------------------------- predict_many
def test_predict_many_equals_single_predicts_in_one_engine_call(home, monkeypatch):
    s = session_for()
    cands = [c.config.to_dict() for c in s.rec.candidates[:18]]
    bad = copy.deepcopy(cands[0])
    bad["settings"]["implement"]["model"] = "gpt-0-unknown"
    new = edited(cands[1])
    configs = [*cands, new, bad, copy.deepcopy(cands[2])]  # a new one, an invalid one, a repeat
    calls: list[int] = []
    real = engine.predict_with_medians

    def counted(belief, task, cfgs, *a, **k):
        calls.append(len(cfgs))
        return real(belief, task, cfgs, *a, **k)

    monkeypatch.setattr(engine, "predict_with_medians", counted)
    many = P.predict_many(s, {"configs": configs})
    assert many["ok"] is True and many["errors"] == [] and len(many["results"]) == len(configs)
    assert calls == [19]  # every valid distinct configuration in one call
    fresh = session_for()
    singles = [P.predict(fresh, {"config": c}) for c in configs]
    assert json.loads(json.dumps(many["results"])) == json.loads(json.dumps(singles))
    assert [r["ok"] for r in many["results"]] == [True] * 19 + [False, True]
    assert many["results"][-2]["errors"][0]["piece"] == "implement"
    assert many["results"][-1] == many["results"][2]
    calls.clear()  # all cached now: no engine call
    again = P.predict_many(s, {"configs": configs})
    assert calls == [] and again == many


def test_predict_many_rejects_a_wrong_body(home):
    s = session_for()
    for body in (None, {}, {"configs": "x"}, {"config": {}}):
        out = P.predict_many(s, body)
        assert out["ok"] is False and out["results"] == [] and out["errors"][0]["piece"] is None
    out = P.predict_many(s, {"configs": [{}] * (P.MAX_MANY + 1)})
    assert out["ok"] is False and str(P.MAX_MANY) in out["errors"][0]["message"]
    assert P.predict_many(s, {"configs": []}) == {"ok": True, "errors": [], "results": []}


# ---------------------------------------------------------------- bands
def nested(bands: dict) -> None:
    assert set(bands) >= set(BANDS), sorted(bands)
    for name in BANDS:
        b = bands[name]
        assert list(b) == list(LEVELS), (name, b)
        for inner, outer in zip(LEVELS, LEVELS[1:]):
            (ilo, ihi), (olo, ohi) = b[inner], b[outer]
            assert olo <= ilo <= ihi <= ohi, (name, inner, outer, b)


def test_bands_are_nested_everywhere_and_within_follows_chance(home, capsys):
    obj, _ = recommend_json(capsys)
    s = session_for()
    ctx = context_payload(s)
    seen = [c["numbers"]["bands"] for c in ctx["candidates"]] + [ch["bands"] for ch in ctx["choices"]]
    seen.append(ctx["reference"]["numbers"]["bands"])
    out = P.predict(s, {"config": edited(ctx["candidates"][0]["config"])})
    seen.append(out["numbers"]["bands"])
    for bands in seen:
        nested(bands)
        for lv in LEVELS:  # p_accepted_within = g + (1 - g) p_fix, end by end
            lo, hi = bands["chance"][lv]
            assert bands["p_accepted_within"][lv] == [engine.r6(accepted_within(lo, s.rec.rescue)),
                                                      engine.r6(accepted_within(hi, s.rec.rescue))]
    # the other three bands are recommend's own
    for ch, want in zip(ctx["choices"], obj["choices"]):
        assert as_recommend(ch["bands"]) == want["bands"]


# ---------------------------------------------------------------- pieces, rounds, errors
def test_pieces_carry_gate_pass_and_cost_ranges_and_the_reply_rounds(home):
    s = session_for()
    usual = s.rec.usual.config  # implement_review, a gate after the review
    gated = {g.after for g in usual.workflow.control.gates}
    assert gated
    out = P.predict(s, {"config": usual.to_dict()})
    assert out["ok"]
    for p in out["pieces"]:
        c = p["run_cost_usd"]
        assert set(c) == {"mean", "lo", "hi"} and c["lo"] <= c["mean"] <= c["hi"]
        if p["piece"] in gated:
            g = p["gate_pass"]
            assert set(g) == {"mean", "lo", "hi"} and 0 <= g["lo"] <= g["mean"] <= g["hi"] <= 1, g
        else:
            assert p["gate_pass"] is None
    r = out["rounds"]
    assert set(r) == {"mean", "lo", "hi", "max"} and r["max"] == usual.workflow.control.budget_rounds
    assert r["lo"] <= r["mean"] <= r["hi"]
    pred = s.rec.usual.prediction
    assert r["mean"] == engine.r6(pred.rounds.mean)
    review = next(p for p in out["pieces"] if p["piece"] in gated)
    assert review["gate_pass"]["mean"] == engine.r6(pred.per_piece[review["piece"]].gate_pass.mean)


def test_errors_and_warnings_name_their_piece(home):
    s = session_for()
    good = s.rec.usual.config.to_dict()
    gate = good["workflow"]["control"]["gates"][0]
    cases = []
    bad = copy.deepcopy(good)
    bad["workflow"]["edges"].append(["implement", "nowhere"])
    cases.append((bad, "'nowhere' is not a piece or an artifact", "implement"))
    bad = copy.deepcopy(good)
    bad["workflow"]["control"]["gates"][0]["on_fail"] = "nowhere"
    cases.append((bad, "on_fail 'nowhere'", gate["after"]))
    bad = copy.deepcopy(good)
    bad["settings"]["review"]["model"] = ""
    cases.append((bad, "setting 'review' has no model", "review"))
    bad = copy.deepcopy(good)
    bad["workflow"]["pieces"][1]["width"] = "two"
    cases.append((bad, "width must be a whole number", good["workflow"]["pieces"][1]["id"]))
    bad = copy.deepcopy(good)
    bad["workflow"]["control"]["budget_rounds"] = 0
    cases.append((bad, "control.budget_rounds", None))
    bodies = [c for c, _, _ in cases]
    many = P.predict_many(s, {"configs": bodies})["results"]
    for (cfg, words, piece), out in zip(cases, many):
        assert out["ok"] is False and out["errors"], words
        assert all(set(e) == {"piece", "message"} for e in out["errors"])
        assert any(words in e["message"] and e["piece"] == piece for e in out["errors"]), (words, out["errors"])
    for message, want in (("piece 'review' has two gates", "review"), ("setting for 'x', which is not a piece", None),
                          ("gate id 'g' appears twice", None), ("'review' is both a piece and an artifact", "review"),
                          ("a workflow needs at least one piece", None)):
        assert P.piece_of(message, s.rec.usual.config) == want, message


# ---------------------------------------------------------------- the catalog: runs behind
def test_runs_behind_counts_the_recorded_runs_and_sums(home):
    runs = [H.USUAL] * 3 + [CATALOG[0]] * 2 + [CATALOG[-1]]
    write_runs(home, runs)
    s = session_for()
    cat = context_payload(s)["catalog"]
    want_total: dict[str, int] = {}
    want_role: dict[str, dict[str, int]] = {}
    for cfg in runs:
        for m in {st.model for st in cfg.settings.values()}:
            want_total[m] = want_total.get(m, 0) + 1
        for p in cfg.workflow.pieces:
            want_role.setdefault(cfg.settings[p.id].model, {}).setdefault(p.role, 0)
        for m, r in {(cfg.settings[p.id].model, p.role) for p in cfg.workflow.pieces}:
            want_role[m][r] += 1
    for m in cat["models"]:
        rb = m["runs_behind"]
        assert set(rb) == {"total", "by_role", "by_effort"}
        assert rb["total"] == want_total.get(m["id"], 0)
        assert rb["by_role"] == want_role.get(m["id"], {})
        assert set(rb["by_effort"]) == set(rb["by_role"])
        for role, n in rb["by_role"].items():
            assert sum(rb["by_effort"][role].values()) == n  # one effort per model and role in these runs
        assert rb["total"] <= sum(rb["by_role"].values())
    totals = [m["runs_behind"]["total"] for m in cat["models"]]
    assert totals == sorted(totals, reverse=True)  # most runs first
    roles = {r["id"]: r["runs"] for r in cat["roles"]}
    assert roles["tester"] == 0
    for role in {p.role for cfg in runs for p in cfg.workflow.pieces}:
        assert roles[role] == sum(1 for cfg in runs if role in {p.role for p in cfg.workflow.pieces})


# ---------------------------------------------------------------- the context, --start and --rec
def test_context_names_the_goal_the_rescue_text_and_the_options(home, capsys):
    obj, _ = recommend_json(capsys)
    ctx = context_payload(session_for())
    goal = next(ch for ch in ctx["choices"] if ch["key"] == "goal")
    assert ctx["goal_config_id"] == goal["config"] == obj["goal"]["config"]
    assert isinstance(ctx["rescue"]["text"], str) and ctx["rescue"]["text"] == obj["rescue"]["text"]
    assert [(c["option"], c["key"]) for c in ctx["choices"]] == [(c["option"], c["key"]) for c in obj["choices"]]
    assert [c["option"] for c in ctx["choices"]] == list(range(1, len(ctx["choices"]) + 1))


def test_start_takes_any_option_the_reference_or_a_candidate(home, capsys):
    obj, _ = recommend_json(capsys)
    s = session_for()
    ids = [c["config"] for c in obj["choices"]] + [obj["reference"]["config"]["id"], s.rec.candidates[-1].config.id]
    for cid in dict.fromkeys(ids):
        st = session_for("--start", cid)
        assert st.start is not None and st.start.id == cid
        assert context_payload(st)["start"]["id"] == cid


def test_rec_with_start_from_each_option(home, capsys, monkeypatch, tmp_path_factory):
    obj, _ = recommend_json(capsys)
    import loopmath.belief.fit as bf

    fs = H.synth_fit(tmp_path_factory)["state"]
    monkeypatch.setattr(bf, "load_fit", lambda h, fit_id: fs)
    (home / "fits" / obj["fit"]["id"]).mkdir(parents=True)
    for ch in obj["choices"]:
        s, code = build_session(parse(["builder", "--rec", obj["rec"], "--start", ch["config"], "--no-open"]))
        assert code == 0 and s.rec_id == obj["rec"] and s.start.id == ch["config"]
        ctx = context_payload(s)
        assert ctx["rec"] == obj["rec"] and ctx["goal_config_id"] == obj["goal"]["config"]
        assert [(c["option"], c["key"], c["config"], c["cost_per_accepted_usd"]) for c in ctx["choices"]] == \
            [(c["option"], c["key"], c["config"], c["cost_per_accepted_usd"]) for c in obj["choices"]]


# ---------------------------------------------------------------- the server
def test_server_answers_predict_many(home):
    s = session_for()
    srv, thread = srv_mod.start(s)
    port = srv.server_address[1]
    try:
        _, _, body = request(port, "GET", "/api/context")
        ctx = json.loads(body)
        configs = [c["config"] for c in ctx["candidates"][:5]]
        code, ctype, body = request(port, "POST", "/api/predict_many", json.dumps({"configs": configs}).encode())
        out = json.loads(body)
        assert code == 200 and ctype.startswith("application/json") and out["ok"]
        assert [r["config_id"] for r in out["results"]] == [c["id"] for c in configs]
        assert [r["numbers"]["cost_per_accepted_usd"] for r in out["results"]] == \
            [c["numbers"]["cost_per_accepted_usd"] for c in ctx["candidates"][:5]]
        code, _, body = request(port, "POST", "/api/predict_many", b"[1")
        assert code == 400 and json.loads(body)["errors"][0]["piece"] is None
        code, _, body = request(port, "POST", "/api/nope", b"{}")
        assert code == 404 and json.loads(body)["errors"] == [{"piece": None, "message": "no such path: /api/nope"}]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


@pytest.mark.parametrize("n", [20])
def test_twenty_one_step_neighbours_answer_quickly(home, n):
    """The page's next steps: every one-step change of a build, re-ranked after each edit."""
    import time

    s = session_for()
    base = s.rec.usual.config.to_dict()
    efforts = ["low", "medium", "high", "xhigh"]
    neighbours = []
    for w in range(1, n + 1):
        d = edited(base, width=w)
        d["settings"]["implement"]["effort"] = efforts[w % len(efforts)]
        neighbours.append(d)
    t0 = time.perf_counter()
    out = P.predict_many(s, {"configs": neighbours})
    dt = time.perf_counter() - t0
    assert out["ok"] and len(out["results"]) == n
    assert dt < 5.0, dt  # the synthetic fit; the 1 s target is checked on the RQ1 store copy


# ---------------------------------------------------------------- items 9 to 11: only the current models
def uses_only(cfg: dict, models) -> bool:
    return all(s["model"] in set(models) for s in cfg["settings"].values())


def test_default_models_retire_the_rest_in_recommend_and_the_builder(home, capsys, monkeypatch):
    """With neither --models nor models.allowed, only DEFAULT_MODELS are offered. The store's usual runs on
    made-up models: it stays the reference, labelled a retired model, and is never a pick, choice or candidate."""
    import loopmath.workflows.candidates as wc
    from loopmath import cli
    from loopmath.recommend.commands import DEFAULT_MODELS
    from loopmath.types import Setting

    (home / "config.toml").unlink()
    usual_id = H.USUAL.id
    # the synthetic generator ignores `allowed`: with nothing offered, recommend says so and exits 1
    assert cli.main(["recommend", *TASK, "--json"]) == 1
    assert "no candidate workflow uses only the offered models" in capsys.readouterr().err
    opus, astra = Setting("claude-code", "claude-opus-5-5", "high"), Setting("codex", "gpt-6-astra", "xhigh")
    current = [H.make_config(H.SD.SOLO, {"implement": opus}), H.make_config(H.SD.SOLO, {"implement": astra}),
               H.make_config(H.SD.IR, {"implement": astra, "review": opus})]
    synthetic = wc.candidates
    monkeypatch.setattr(wc, "candidates", lambda *a, **k: [*synthetic(*a, **k), *((c, "catalog") for c in current)])
    obj, stored = recommend_json(capsys)
    ref = obj["reference"]
    assert ref["config"]["id"] == usual_id and ref["retired_models"] == sorted({s.model for s in H.USUAL.settings.values()})
    assert ref["label"].endswith("(retired model)")
    assert obj["choices"] and all(c["config"] != usual_id and c["key"] != "reference" for c in obj["choices"])
    by_id = {c["config"]["id"]: c for c in stored["candidates"]}
    for ch in obj["choices"]:
        for member in ch["members"]:
            assert uses_only(by_id[member]["config"], DEFAULT_MODELS), member
    assert obj["goal"]["config"] != usual_id and uses_only(by_id[obj["goal"]["config"]]["config"], DEFAULT_MODELS)
    assert "(retired model)" in ref["text"]  # the plan text names it where it shows the reference
    assert cli.main(["recommend", *TASK]) == 0
    assert "(retired model)" in capsys.readouterr().out
    assert all(uses_only(c["config"], DEFAULT_MODELS) for c in stored["candidates"] if c["config"]["id"] != usual_id)
    # the builder: the catalog offers the list, the reference is marked, no candidate or choice is retired
    s = session_for()
    ctx = context_payload(s)
    assert [m["id"] for m in ctx["catalog"]["models"]] == [m for m in DEFAULT_MODELS] or \
        sorted(m["id"] for m in ctx["catalog"]["models"]) == sorted(DEFAULT_MODELS)
    assert ctx["reference"]["retired_models"] == ref["retired_models"]
    assert all(uses_only(c["config"], DEFAULT_MODELS) for c in ctx["candidates"])
    assert all(uses_only(ch["configuration"], DEFAULT_MODELS) for ch in ctx["choices"])
    # the retired usual cannot be predicted as a pick: each retired piece says so
    out = P.predict(s, {"config": H.USUAL.to_dict()})
    assert out["ok"] is False
    assert {e["piece"] for e in out["errors"]} == {p.id for p in H.USUAL.workflow.pieces}
    assert all("retired" in e["message"] for e in out["errors"])
    # --models names any model again
    assert cli.main(["recommend", *TASK, "--json", "--models", ",".join(dict.fromkeys(
        st.model for st in H.SETTINGS))]) == 0
    again = json.loads(capsys.readouterr().out)
    assert "retired_models" not in again["reference"]


def test_models_allowed_in_config_is_what_the_builder_offers(home):
    from test_builder_api import SYNTH_MODELS, allow_models

    allow_models(home, SYNTH_MODELS[:3])
    ctx = context_payload(session_for())
    assert sorted(m["id"] for m in ctx["catalog"]["models"]) == sorted(SYNTH_MODELS[:3])
    assert all(uses_only(c["config"], SYNTH_MODELS[:3]) for c in ctx["candidates"])


# ---------------------------------------------------------------- one bad entry never sinks a batch (22R B1)
def gate_after(cfg: dict, after) -> dict:
    d = copy.deepcopy(cfg)
    d["workflow"]["control"]["gates"] = [{"id": "test_gate", "after": after, "rule": "review_approve",
                                          "on_fail": d["workflow"]["pieces"][0]["id"]}]
    return d


@pytest.mark.parametrize("after", [[], {}, 3, None])
def test_a_gate_after_that_is_not_a_piece_id_is_a_structured_error(home, after):
    s = session_for()
    good = [c.config.to_dict() for c in s.rec.candidates[:2]]
    bad = gate_after(good[0], after)
    one = P.predict(s, {"config": bad})
    assert one["ok"] is False and one["config_id"] is None
    assert any("gates[0]" in e["message"] and e["piece"] is None for e in one["errors"]), one["errors"]
    many = P.predict_many(s, {"configs": [good[0], bad, good[1]]})
    assert many["ok"] is True and [r["ok"] for r in many["results"]] == [True, False, True]
    assert many["results"][1]["errors"] == one["errors"]
    assert [r["config_id"] for r in many["results"][::2]] == [c["id"] for c in good]


def test_mixed_batch_over_http_keeps_the_valid_answers(home):
    s = session_for()
    srv, thread = srv_mod.start(s)
    port = srv.server_address[1]
    try:
        good = [c.config.to_dict() for c in s.rec.candidates[:2]]
        body = {"configs": [good[0], gate_after(good[0], []), gate_after(good[1], {}), good[1]]}
        code, _, raw = request(port, "POST", "/api/predict_many", json.dumps(body).encode())
        out = json.loads(raw)
        assert code == 200 and out["ok"] and [r["ok"] for r in out["results"]] == [True, False, False, True]
        code, _, raw = request(port, "POST", "/api/predict", json.dumps({"config": gate_after(good[0], [])}).encode())
        assert code == 200 and json.loads(raw)["ok"] is False
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def test_an_engine_failure_on_one_configuration_fails_only_that_one(home, monkeypatch):
    s = session_for()
    good = [c.config.to_dict() for c in s.rec.candidates[:3]]
    doomed = good[1]["id"]
    real = P.answer_all

    def fragile(session, cfgs):
        if any(c.id == doomed for c in cfgs):
            raise RuntimeError("no rows for this one")
        return real(session, cfgs)

    monkeypatch.setattr(P, "answer_all", fragile)
    out = P.predict_many(s, {"configs": good})
    assert [r["ok"] for r in out["results"]] == [True, False, True]
    assert out["results"][1]["config_id"] == doomed and "no rows for this one" in out["results"][1]["errors"][0]["message"]
    assert doomed not in s.cache and good[0]["id"] in s.cache
    # 22R B2: the same batch again, where the failing one is the only configuration not cached
    again = P.predict_many(s, {"configs": good})
    assert again["results"] == out["results"]
    # one cached good configuration and one new failing one
    fresh = edited(good[1], width=3)
    doomed = P.parse_config(fresh, P.known_models(s))[0].id
    mixed = P.predict_many(s, {"configs": [good[0], fresh]})
    assert [r["ok"] for r in mixed["results"]] == [True, False]
    assert mixed["results"][0] == out["results"][0] and mixed["results"][1]["config_id"] == doomed
    # and alone: `/api/predict` answers the structured error, not a 500
    alone = P.predict(s, {"config": fresh})
    assert alone["ok"] is False and "no rows for this one" in alone["errors"][0]["message"] and doomed not in s.cache


def test_every_option_workflow_is_in_the_context_whatever_the_cutoff(home, monkeypatch):
    """22R B3: the pair's exploration workflow ranks below the page's cutoff, yet the context still carries it, as it
    does every option's workflows, so the page can show and start every option."""
    from loopmath.builder import context as ctx_mod

    s = session_for()
    order = [c.config.id for c in s.rec.candidates]
    pair = next(ch for ch in s.rec.payload()["choices"] if ch["key"] == "pair")
    cut = order.index(pair["explore_config"])
    assert cut >= 1
    monkeypatch.setattr(ctx_mod, "TOP_CANDIDATES", cut)  # the exploration workflow is the first one cut
    ctx = context_payload(s)
    have = {c["config"]["id"] for c in ctx["candidates"]}
    assert order[cut] not in {c["config"]["id"] for c in ctx["candidates"][:cut]}
    for ch in ctx["choices"]:
        for cid in (ch["config"], *ch["members"], ch.get("explore_config")):
            assert cid is None or cid in have, (ch["key"], cid)
        # what the page resolves for the option (the pair's exploration workflow, else the option's)
        assert (ch.get("explore_config") or ch["config"]) in have
    by_id = {c["config"]["id"]: c for c in ctx["candidates"]}
    assert by_id[pair["explore_config"]]["numbers"] == P.numbers_of(s.rec, s.rec.by_id(pair["explore_config"]).prediction)
