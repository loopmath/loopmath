"""`loopmath builder` (spec 02, `builder`): the context, predictions that equal `recommend`'s, configuration
checks and ids, and the server on 127.0.0.1. The fit is the search tests' synthetic one (no real data); the
only network is 127.0.0.1."""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recommend"))

import search_synth as H  # noqa: E402

from loopmath import cli, cli_registry  # noqa: E402
from loopmath.builder import server as srv_mod  # noqa: E402
from loopmath.builder.context import build_session, context_payload  # noqa: E402
from loopmath.builder.predict import predict  # noqa: E402
from loopmath.recommend import storeread  # noqa: E402
from loopmath.workflows.ids import config_id  # noqa: E402
from loopmath.workflows.ocp import configuration_from_any  # noqa: E402

S0, S1, S2, S3, S4, S5 = H.SETTINGS
CATALOG = [H.make_config(H.SD.SOLO, {"implement": s}) for s in H.SETTINGS]
CATALOG += [H.make_config(H.SD.IR, {"implement": a, "review": b}) for a in (S0, S4) for b in (S1, S5)]
TASK = ["--type", "feature", "--repo", "acme/api"]
SYNTH_MODELS = list(dict.fromkeys(s.model for s in H.SETTINGS))  # made-up models: offered through config (0.2.2)


def allow_models(home, models) -> None:
    (home / "config.toml").write_text("[models]\nallowed = [" + ", ".join(json.dumps(m) for m in models) + "]\n")


def write_runs(home, configs) -> None:
    (home / "runs").mkdir(exist_ok=True)
    rows = []
    for i, c in enumerate(configs):
        run = f"run_{i:03d}"
        rows.append({"run": run, "task_type": "feature", "repo": "acme/api", "config": c.id, "source": "live",
                     "started_at": storeread.iso(storeread.now_local()), "state": "finished"})
        (home / "runs" / f"{run}.ocp.json").write_text(json.dumps({"run": {"configuration": c.to_dict()}}))
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def home(tmp_path, tmp_path_factory, monkeypatch):
    fs = H.synth_fit(tmp_path_factory)["state"]
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    import loopmath.belief.state as bs
    import loopmath.workflows.candidates as wc

    def catalog_candidates(task, usual, allowed, user=(), recorded=()):
        return ([(usual, "usual")] + [(c, "catalog") for c in CATALOG if c.id != usual.id]
                + [(r, "recorded") for r in recorded] + [(u, "user") for u in user])

    monkeypatch.setattr(bs, "load_latest", lambda h: fs)
    monkeypatch.setattr(wc, "candidates", catalog_candidates)
    write_runs(home, [H.USUAL] * 3)
    allow_models(home, SYNTH_MODELS)
    return home


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="loopmath")
    cli_registry.register(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(argv)


def session_for(*extra: str):
    s, code = build_session(parse(["builder", *TASK, "--no-open", *extra]))
    assert code == 0 and s is not None
    return s


def recommend_json(capsys, *extra: str) -> tuple[dict, dict]:
    """`recommend --json` and the stored recommendation (it keeps every candidate)."""
    assert cli.main(["recommend", *TASK, "--json", *extra]) == 0
    obj = json.loads(capsys.readouterr().out)
    stored = json.loads((Path(os.environ["LOOPMATH_HOME"]) / "recs" / f"{obj['rec']}.json").read_text())
    return obj, stored


def as_recommend(bands: dict) -> dict:
    """`bands` without the builder's `p_accepted_within` band (0.2.2), as `recommend --json` gives them."""
    return {k: v for k, v in bands.items() if k != "p_accepted_within"}


def without_chance(numbers: dict) -> dict:
    return {k: (as_recommend(v) if k == "bands" else v) for k, v in numbers.items() if k != "chance"}


# ---------------------------------------------------------------- the context
def test_context_has_the_api_shape(home, capsys):
    obj, _ = recommend_json(capsys)
    s = session_for()
    ctx = json.loads(json.dumps(context_payload(s)))
    assert list(ctx) == ["schema", "task", "rule", "fit", "rec", "goal_config_id", "rescue", "reference", "choices",
                         "candidates", "catalog", "start"]
    assert ctx["goal_config_id"] == obj["goal"]["config"]
    assert ctx["schema"] == "loopmath.builder.context/1" and ctx["start"] is None and ctx["rec"] is None
    assert ctx["fit"]["id"] == obj["fit"]["id"] and ctx["fit"]["runs"] > 0
    assert ctx["rule"] == obj["rule"] and ctx["rescue"] == obj["rescue"]
    ref = {**ctx["reference"], "numbers": without_chance(ctx["reference"]["numbers"])}
    assert {k: v for k, v in obj["reference"].items() if k != "prediction"} == ref
    assert [c["config"] for c in ctx["choices"]] == [c["config"] for c in obj["choices"]]
    for ch, want in zip(ctx["choices"], obj["choices"]):
        assert {k: (as_recommend(v) if k == "bands" else v) for k, v in ch.items() if k != "configuration"} == want
        assert ch["configuration"]["id"] == ch["config"]
    cands = ctx["candidates"]
    assert 0 < len(cands) and all(set(c) >= {"config", "label", "numbers", "origin"} for c in cands)
    ells = [c["numbers"]["cost_per_accepted_usd"]["mean"] for c in cands[:60]]
    assert ells == sorted(ells)  # the first 60 by cost per accepted result
    assert H.USUAL.id in {c["config"]["id"] for c in cands}  # the recorded one is kept
    cat = ctx["catalog"]
    assert set(cat) >= {"harnesses", "models", "efforts", "roles", "shapes"}
    models = {m["id"]: m for m in cat["models"]}
    assert {s.model for s in H.SETTINGS} <= set(models)
    used = {s.model for s in H.USUAL.settings.values()}  # the store's three runs all used the usual
    assert all(models[m]["runs_behind"]["total"] == 3 for m in used)
    assert all(m["runs_behind"]["total"] == 0 for m in models.values() if m["id"] not in used)
    assert {"claude-code", "codex"} <= set(cat["harnesses"]) and "xhigh" in cat["efforts"]
    assert all(set(sh) >= {"id", "title", "pieces", "edges", "gates", "workflow"} for sh in cat["shapes"])


# ---------------------------------------------------------------- predict equals recommend
def test_predict_equals_recommend_for_three_candidates(home, capsys):
    obj, stored = recommend_json(capsys)
    s = session_for()
    by_id = {c["config"]["id"]: c for c in stored["candidates"]}
    goal, ref = obj["goal"]["config"], obj["reference"]["config"]["id"]
    third = next(c["config"]["id"] for c in stored["candidates"][3:] if c["config"]["id"] not in (goal, ref))
    for cid in (goal, ref, third):
        want = by_id[cid]
        out = predict(s, {"config": copy.deepcopy(want["config"])})
        assert out["ok"] and out["errors"] == [] and out["config_id"] == cid
        assert without_chance(out["numbers"]) == want["numbers"]
        assert out["label"] == want["label"] and out["known"] is True
        assert out["numbers"]["run_cost_usd"]["median_basis"] == "draws"
        assert [p["piece"] for p in out["pieces"]] == [p["id"] for p in want["config"]["workflow"]["pieces"]]
        assert all(p["run_cost_usd"]["mean"] > 0 for p in out["pieces"])
        assert 1 <= out["rank"]["by_cost_per_accepted"] <= out["rank"]["of"] == len(s.rec.candidates)
    choice = obj["choices"][0]
    assert predict(s, {"config": by_id[choice["config"]]["config"]})["numbers"]["chance"] == choice["chance"]
    first = predict(s, {"config": by_id[goal]["config"]})
    assert first["rank"]["by_cost_per_accepted"] == 1 + sum(
        1 for c in stored["candidates"] if c["config"]["id"] != goal
        and (c["numbers"]["cost_per_accepted_usd"]["mean"] < first["numbers"]["cost_per_accepted_usd"]["mean"]))


def test_predict_caches_by_config_id(home):
    s = session_for()
    cfg = s.rec.candidates[0].config.to_dict()
    predict(s, {"config": cfg})
    assert cfg["id"] in s.cache
    before = s.cache[cfg["id"]]
    assert predict(s, {"config": {**cfg, "id": None}})["numbers"] == before["numbers"] and len(s.cache) == 1


def test_the_question_task_is_keyed_as_recommend_keys_it(home, monkeypatch):
    """The belief sees the question `recommend` asks: keyed by `commands.question_key(belief)` when recommend
    has it (lane 21F's stable seed key), else by the fit id as in 0.2.0."""
    from loopmath.recommend import commands as C

    keys: list[str] = []
    real = C.question_task

    def recorded(task, key, rule, configs):
        keys.append(key)
        return real(task, key, rule, configs)

    monkeypatch.setattr(C, "question_task", recorded)
    monkeypatch.delattr(C, "question_key", raising=False)  # recommend before 21F
    s = session_for()
    assert keys and set(keys) == {s.belief.fit_id}
    keys.clear()
    monkeypatch.setattr(C, "question_key", lambda belief: "seed_" + belief.fit_id, raising=False)
    s2 = session_for()
    assert set(keys) == {"seed_" + s.belief.fit_id} and s2.task.id != s.task.id


def test_two_fits_of_the_same_data_give_the_builder_recommends_numbers(home, monkeypatch, capsys):
    """21F (F8): two fits of the same data share a seed key, so `recommend` asks both the same question. The
    builder, through recommend's own `question_key`, asks it too: same asked task, and each fit's builder
    numbers equal that fit's `recommend --json` and each other's."""
    from loopmath.belief.state import FitState
    from loopmath.recommend import commands as C
    import loopmath.belief.state as bs

    if not hasattr(C, "question_key"):
        pytest.skip("recommend keys its question by the fit id until lane 21F merges")
    fs = bs.load_latest(home)
    twin = FitState(fs.path, meta={**fs.meta, "fit": fs.fit_id + "_twin"})
    assert twin.fit_id != fs.fit_id and C.question_key(twin) == C.question_key(fs) != fs.fit_id
    keys: list[str] = []
    real = C.question_task

    def recorded(task, key, rule, configs):
        keys.append(key)
        return real(task, key, rule, configs)

    monkeypatch.setattr(C, "question_task", recorded)
    seen = []
    for belief in (fs, twin):
        monkeypatch.setattr(bs, "load_latest", lambda h, b=belief: b)
        obj, stored = recommend_json(capsys)
        keys.clear()
        s = session_for()
        assert s.belief is belief and obj["fit"]["id"] == belief.fit_id
        assert keys and set(keys) == {C.question_key(belief)}
        by_id = {c["config"]["id"]: c for c in stored["candidates"]}
        got = {}
        for cid in (obj["goal"]["config"], obj["reference"]["config"]["id"]):
            out = predict(s, {"config": copy.deepcopy(by_id[cid]["config"])})
            assert without_chance(out["numbers"]) == by_id[cid]["numbers"]
            got[cid] = out["numbers"]
        seen.append((s.task.id, got))
    assert seen[0] == seen[1]


# ---------------------------------------------------------------- checks and ids
def test_invalid_configurations_get_readable_errors(home):
    s = session_for()
    good = s.rec.usual.config.to_dict()  # implement_review with a repair gate
    cases = []
    bad = copy.deepcopy(good)
    bad["settings"]["implement"]["model"] = "gpt-0-unknown"
    cases.append((bad, "model 'gpt-0-unknown' is a retired or unknown model", "implement"))
    bad = copy.deepcopy(good)
    bad["workflow"]["pieces"][0]["width"] = 0
    cases.append((bad, "width must be an integer >= 1", good["workflow"]["pieces"][0]["id"]))
    bad = copy.deepcopy(good)
    bad["workflow"]["control"]["gates"][0]["on_fail"] = "nowhere"
    cases.append((bad, "on_fail 'nowhere' is not a piece", good["workflow"]["control"]["gates"][0]["after"]))
    bad = copy.deepcopy(good)
    del bad["settings"]["review"]
    cases.append((bad, "piece 'review' has no setting", "review"))
    bad = copy.deepcopy(good)
    bad["settings"]["review"] = {"harness": "codex"}
    cases.append((bad, "setting 'review' has no model", "review"))
    cases.append(({"settings": {}}, "config.workflow: expected an object", None))
    cases.append(("not a config", "send {\"config\"", None))
    for cfg, words, piece in cases:  # 0.2.2: each error is {piece, message}, tied to the piece it is about
        out = predict(s, {"config": cfg})
        assert out["ok"] is False and any(words in e["message"] and e["piece"] == piece for e in out["errors"]), (
            words, piece, out["errors"])
        assert "numbers" not in out
    assert predict(s, None)["ok"] is False


def test_a_new_configuration_gets_the_id_loopmath_computes(home):
    s = session_for()
    base = s.rec.usual.config
    d = base.to_dict()
    d["workflow"]["pieces"][0]["width"] = 2
    d["settings"]["implement"] = {**S3.to_dict()}
    want = configuration_from_any({**d, "id": None})
    assert want.id == config_id(want.workflow, want.settings) and want.id not in {c.config.id for c in s.rec.candidates}
    for sent in (None, "cfg_000000000000", base.id):
        out = predict(s, {"config": {**d, "id": sent}})
        assert out["ok"] and out["config_id"] == want.id and out["known"] is False
    assert out["rank"]["of"] == len(s.rec.candidates) + 1
    assert out["label"].startswith("implement_review: 2 x claude-fable-9/medium")
    # a model_ref left over from before a model change names another model: dropped, so the id is the content's
    stale = copy.deepcopy(d)
    stale["settings"]["implement"]["model_ref"] = {"raw": S0.model, "id": S0.model}
    assert predict(s, {"config": stale})["config_id"] == want.id
    # a missing harness is filled in from the model, and said so
    d2 = copy.deepcopy(d)
    del d2["settings"]["implement"]["harness"]
    out = predict(s, {"config": d2})
    assert out["ok"] and out["config_id"] == want.id and any(
            "harness claude-code" in w["message"] and w["piece"] == "implement" for w in out["warnings"])


# ---------------------------------------------------------------- start points
def test_rec_and_start(home, capsys, monkeypatch, tmp_path_factory):
    obj, stored = recommend_json(capsys)
    import loopmath.belief.fit as bf

    fs = H.synth_fit(tmp_path_factory)["state"]  # built once, by the fixture
    monkeypatch.setattr(bf, "load_fit", lambda h, fit_id: fs)
    (home / "fits" / obj["fit"]["id"]).mkdir(parents=True)  # the recommendation's fit is still kept
    s, code = build_session(parse(["builder", "--rec", obj["rec"], "--start", obj["goal"]["config"], "--no-open"]))
    assert code == 0 and s.rec_id == obj["rec"] and s.start.id == obj["goal"]["config"]
    ctx = context_payload(s)
    assert ctx["rec"] == obj["rec"] and ctx["start"]["id"] == obj["goal"]["config"]
    assert [(c["config"], c["cost_per_accepted_usd"]) for c in ctx["choices"]] == \
        [(c["config"], c["cost_per_accepted_usd"]) for c in obj["choices"]]
    assert cli.main(["builder", *TASK, "--start", "cfg_ffffffffffff", "--no-open"]) == 2
    assert "cfg_ffffffffffff" in capsys.readouterr().err
    assert cli.main(["builder", "--rec", "rec_nope", "--no-open"]) == 2


# ---------------------------------------------------------------- the server
def request(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, str, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read()
    finally:
        conn.close()


def test_server_binds_127_0_0_1_answers_and_stops_cleanly(home):
    s = session_for()
    srv, thread = srv_mod.start(s)
    port = srv.server_address[1]
    try:
        assert srv.server_address[0] == "127.0.0.1" and port > 0 and srv.url == f"http://127.0.0.1:{port}/"
        code, ctype, body = request(port, "GET", "/")
        assert code == 200 and ctype.startswith("text/html") and b"/api/predict" in body
        code, ctype, body = request(port, "GET", "/assets/base.css")
        assert code == 200 and ctype.startswith("text/css")
        assert request(port, "GET", "/assets/../../x.css")[0] == 404
        code, ctype, body = request(port, "GET", "/api/context")
        ctx = json.loads(body)
        assert code == 200 and ctype.startswith("application/json") and ctx["schema"] == "loopmath.builder.context/1"
        cfg = ctx["candidates"][0]["config"]
        code, _, body = request(port, "POST", "/api/predict", json.dumps({"config": cfg}).encode())
        out = json.loads(body)
        assert code == 200 and out["ok"] and out["numbers"]["cost_per_accepted_usd"] == \
            ctx["candidates"][0]["numbers"]["cost_per_accepted_usd"]
        code, _, body = request(port, "POST", "/api/predict", b"{not json")
        assert code == 400 and json.loads(body)["ok"] is False
        assert request(port, "POST", "/api/other", b"{}")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)
    assert not thread.is_alive()
    with pytest.raises(OSError):
        request(port, "GET", "/api/context")


def test_command_prints_the_url_opens_the_browser_unless_told_and_stops_on_ctrl_c(home, capsys, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(srv_mod.webbrowser, "open", lambda url: opened.append(url))

    def interrupted(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(srv_mod.BuilderServer, "serve_forever", interrupted)
    assert cli.main(["builder", *TASK, "--no-open"]) == 0
    out = capsys.readouterr().out
    assert "loopmath builder: http://127.0.0.1:" in out and "stopped" in out and opened == []
    assert cli.main(["builder", *TASK]) == 0
    url = capsys.readouterr().out.split("loopmath builder: ", 1)[1].split()[0]
    assert opened == [url] and url.startswith("http://127.0.0.1:")
    assert cli.main(["builder", *TASK, "--port", "70000", "--no-open"]) == 1


def test_the_real_command_serves_and_stops_on_sigint(tmp_path, tmp_path_factory):
    """`python -m loopmath builder` on a fitted store, stopped as Ctrl+C stops it."""
    fit_home = H.synth_fit(tmp_path_factory)["home"]
    env = {**os.environ, "LOOPMATH_HOME": str(fit_home), "LOOPMATH_CACHE_DIR": str(tmp_path / "cache"),
           "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    proc = subprocess.Popen([sys.executable, "-m", "loopmath", "builder", *TASK, "--no-open"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line.startswith("loopmath builder: http://127.0.0.1:"), (line, proc.stderr.read() if proc.poll() else "")
        port = int(line.split("http://127.0.0.1:", 1)[1].split("/", 1)[0])
        code, _, body = request(port, "GET", "/api/context")
        assert code == 200 and json.loads(body)["candidates"]
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0 and "stopped" in out, err
    with pytest.raises(OSError):
        request(port, "GET", "/api/context")
