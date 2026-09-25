"""`posterior --subtype S --feature K=V`, as `recommend`: predictions for that task (FINDINGS I14).

The flags go into the task the workflow graph is predicted for, and into the view's `task`
block, which the page reads. The page itself is 0.2.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from loopmath import cli
from loopmath.belief import fit as F
from loopmath.views import posterior as P

from tests.belief import simdata

NOW = datetime.fromisoformat("2026-09-24T17:00:00-07:00")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    home = tmp_path_factory.mktemp("flags") / "home"
    docs, _ = simdata.simulate(160, seed=43, source="live", prefix="run_fl")
    (home / "runs").mkdir(parents=True)
    for doc in docs:
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    F.fit(home, no_prior=True, now=NOW)
    return home, docs[0]["run"]["configuration"]["id"]


def _view(capsys, home, cfg, *extra):
    argv = ["posterior", "--home", str(home), "--type", "feature", "--repo", "acme/api", "--workflow", cfg,
            "--level", "model", "--json", *extra]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_features_reach_the_prediction_and_the_task_block(store, capsys):
    home, cfg = store
    small, _ = _view(capsys, home, cfg, "--feature", "size=s")
    large, _ = _view(capsys, home, cfg, "--feature", "size=l", "--feature", "has_tests=yes")
    assert small["task"]["features"] == {"size": "s"}
    assert large["task"]["features"] == {"size": "l", "has_tests": "yes"}
    cost = [v["workflow"]["prediction"]["cost"]["usd"]["mean"] for v in (small, large)]
    assert cost[0] != pytest.approx(cost[1], rel=1e-6)


def test_subtype_reaches_the_task_and_an_unseen_one_is_noted(store, capsys):
    home, cfg = store
    view, err = _view(capsys, home, cfg, "--subtype", "hot/path")
    assert view["task"]["subtype"] == "hot/path" and view["task"]["from"] == "arguments"
    assert "has no runs of subtype hot/path for feature in acme/api" in err


def test_the_terminal_names_the_task(store, capsys):
    home, cfg = store
    assert cli.main(["posterior", "--home", str(home), "--type", "feature", "--repo", "acme/api", "--workflow", cfg,
                     "--subtype", "hot/path", "--feature", "size=l", "--level", "model"]) == 0
    out = capsys.readouterr().out
    assert "for a feature task in acme/api (hot/path; size=l)" in out


def test_the_page_gets_the_task(store, capsys, tmp_path):
    home, cfg = store
    page = tmp_path / "p.html"
    assert cli.main(["posterior", "--home", str(home), "--type", "feature", "--repo", "acme/api", "--workflow", cfg,
                     "--subtype", "hot/path", "--feature", "size=l", "--html", str(page)]) == 0
    data = P.extract_data(page.read_text())
    assert data["task"]["subtype"] == "hot/path" and data["task"]["features"] == {"size": "l"}


def test_feature_parsing(capsys, store):
    assert P.task_features(["size=l", " Lang = CPP "]) == ({"size": "l", "lang": "cpp"}, [])
    feats, unknown = P.task_features(["problem=heuristic"])
    assert feats == {"extra:problem": "heuristic"} and unknown == ["problem"]
    with pytest.raises(P.ViewError, match="--feature takes K=V"):
        P.task_features(["size"])
    home, cfg = store
    _, err = _view(capsys, home, cfg, "--feature", "problem=heuristic")
    assert "does not model the feature problem" in err
    assert cli.main(["posterior", "--home", str(home), "--feature", "size"]) == 1


def test_choose_task_keeps_subtype_and_features(tmp_path):
    task, source = P.choose_task(tmp_path, "bug_fix", "acme/api", "flaky", {"size": "s"})
    assert (task.subtype, task.features, source) == ("flaky", {"size": "s"}, "arguments")
    task, source = P.choose_task(tmp_path, None, None, "flaky")
    assert (task.type, task.subtype, source) == ("feature", "flaky", "default")
