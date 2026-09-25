"""`loopmath recommend --json` with the workflow search running (spec 05 section 1a, spec 02 section 2): a
synthetic fit the search can read, so `search` is in the output and in the stored recommendation, and the
search's finds are candidates."""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.recommend import storeread

import search_synth as H

S0, S1, S2, S3, S4, S5 = H.SETTINGS
CATALOG = [H.make_config(H.SD.SOLO, {"implement": s}) for s in H.SETTINGS]
CATALOG += [H.make_config(H.SD.IR, {"implement": a, "review": b}) for a in (S0, S4) for b in (S1, S5)]


def write_runs(home, configs) -> None:
    """Index rows and run documents for the usual's history."""
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
    # 0.2.2: the synthetic fit's made-up models are offered through config, not retired by the default list
    models = ", ".join(json.dumps(m) for m in dict.fromkeys(st.model for st in H.SETTINGS))
    (home / "config.toml").write_text(f"[models]\nallowed = [{models}]\n")
    return home


def test_recommend_json_shows_the_search(home, capsys):
    code = cli.main(["recommend", "--type", "feature", "--repo", "acme/api", "--json"])
    obj = json.loads(capsys.readouterr().out)
    assert code == 0 and obj["schema"] == "loopmath.recommend/2"
    assert obj["usual"] is not None and obj["usual"]["config"]["id"] == H.USUAL.id
    keys = list(obj)
    assert keys.index("search") > keys.index("choices")  # after the /2 keys, so earlier readers keep their order
    s = obj["search"]
    assert s is not None and s["method"] == "exact_front+thompson" and s["exact"] is True
    assert s["space"]["settings"] == 6 and s["space"]["configurations"] > 1000 and s["front"]
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    assert stored["search"] == s
    origins = {c["origin"] for c in stored["candidates"]}
    assert origins & {"front", "thompson", "polish"} and "catalog" not in origins
    assert all(set(c["search"]) == {"on_front", "wins"} for c in stored["candidates"])
    assert obj["goal"]["strategy"] is None or obj["goal"]["strategy"]["text"] in obj["message"]
