"""Declared features with few tasks (lane 2C, rule M4; spec 04 section 1). Synthetic runs only.

Four tasks laid out like the four RQ1 phase 2 problems: `output` and `grid` cross 2 by 2,
`input_items` splits the tasks exactly as `output` does, `lang` is the same everywhere and
`domain` rests on one task.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import simdata

from loopmath import cli
from loopmath.belief import fit as F
from loopmath.belief import state as S
from loopmath.belief.commands import _print_summary, _summary
from loopmath.belief.design import task_features
from loopmath.belief.features import FeatureTally, features_line
from loopmath.store.config import Config
from loopmath.taskmodel import FeatureSet
from loopmath.types import Task

NOW = datetime.fromisoformat("2026-09-24T18:30:00-07:00")
DECLARED = {
    "output": {"values": ["structure", "actions"]},
    "grid": {"kind": "bool"},
    "input_items": {"kind": "number", "edges": [1000], "labels": ["small", "large"]},
    "domain": {"values": ["web", "cli"]},
}
TRAITS = {  # task index -> features, as ahc024, ahc026, ahc039, ahc046
    0: {"output": "structure", "grid": "yes", "input_items": 2500, "lang": "cpp", "domain": "web"},
    1: {"output": "actions", "grid": "no", "input_items": 200, "lang": "cpp"},
    2: {"output": "structure", "grid": "no", "input_items": 10000, "lang": "cpp"},
    3: {"output": "actions", "grid": "yes", "input_items": 400, "lang": "cpp"},
}


def _tid(j: int) -> str:
    return f"tsk_sim{j:04d}"


def _docs(n=120, seed=61):
    docs, _ = simdata.simulate(n, seed=seed, source="live", prefix="run_fs", n_tasks=4)
    for d in docs:
        task = d["run"]["task"]
        task["features"] = dict(TRAITS[int(task["id"][-4:])])
    return docs


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    home = tmp_path_factory.mktemp("features") / "home"
    (home / "runs").mkdir(parents=True)
    for doc in _docs():
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    conf = Config(home / "config.toml")
    for key, table in DECLARED.items():
        conf.set(f"features.{key}", table)
    conf.save()
    path = F.fit(home, no_prior=True, now=NOW)
    return home, path, S.load(path)


def test_tally_admits_aliases_and_drops():
    fs = FeatureSet.from_config(DECLARED)
    tally = FeatureTally(fs)
    for j, feats in TRAITS.items():
        task = Task(id=_tid(j), type="feature", repo="ale-bench", features={k: str(v) for k, v in feats.items()})
        tally.add(task, task_features(task, fs))
    nodes, report = tally.admit()
    assert report["admitted"] == {"output": ["actions", "structure"], "grid": ["no", "yes"]}
    assert report["dropped"] == {"lang": "constant", "input_items": "alias of output", "domain": "below min_tasks"}
    assert report["support"]["domain"] == {"web": {"tasks": 1, "repos": 1}}
    assert report["absorbed"] == {"lang": "cpp"}
    assert report["aliases"] == {"input_items": {"of": "output", "map": {"large": "structure", "small": "actions"}}}
    assert nodes == {"feature:output=actions", "feature:output=structure", "feature:grid=no", "feature:grid=yes"}
    assert features_line(report) == ("features: output 2 values (2+2 tasks), grid 2 values (2+2 tasks); "
                                     "dropped lang (constant), input_items (alias of output), domain (below min_tasks)")
    assert tally.task_values(report["admitted"])[_tid(0)] == {"output": "structure", "grid": "yes"}
    fs3 = FeatureSet.from_config({**DECLARED, "min_tasks": 3})
    tally3 = FeatureTally(fs3)
    for j, feats in TRAITS.items():
        task = Task(id=_tid(j), type="feature", repo="r", features={k: str(v) for k, v in feats.items()})
        tally3.add(task, task_features(task, fs3))
    assert tally3.admit()[1]["admitted"] == {}  # min_tasks 3: no value has three tasks, lang is still constant


def test_fit_uses_admitted_values_only(fitted):
    _, _, st = fitted
    fmeta = st.meta["features"]
    assert fmeta["admitted"] == {"output": ["actions", "structure"], "grid": ["no", "yes"]}
    assert fmeta["dropped"]["input_items"] == "alias of output"
    assert fmeta["digest"] == FeatureSet.from_config(DECLARED).digest()
    assert st.features.digest() == fmeta["digest"]  # predictions read features as the fit did
    for head in st.heads.values():
        feats = {n for n in head.index if n.startswith("feature:")}
        assert feats <= {"feature:output=actions", "feature:output=structure", "feature:grid=no", "feature:grid=yes"}
    assert "feature:grid=yes" in st.heads["cost"].index
    assert st.design["task_features"][_tid(0)] == {"output": "structure", "grid": "yes"}


def _task_part(st, task, head="cost"):
    return st.heads[head].eval_parts([st.task_terms(task)])


def test_known_task_gets_no_unseen_feature_node(fitted):
    _, _, st = fitted
    base = Task(id=_tid(0), type="feature", repo="acme/api", features={"output": "structure", "grid": "yes"})
    tid = _tid(0)
    known = Task(id=tid, type=simdata.TYPES[0], repo=simdata.REPOS[0], features=base.features)
    with_extra = Task(id=tid, type=simdata.TYPES[0], repo=simdata.REPOS[0],
                      features={**base.features, "domain": "web", "input_items": "large"})
    mu0, _, _, s2_0 = _task_part(st, known)
    mu1, _, _, s2_1 = _task_part(st, with_extra)
    assert s2_0[0] == 0 and s2_1[0] == 0 and mu0[0] == pytest.approx(mu1[0])
    assert "domain=web: 1 task, not used yet; the task's own node covers it" in st.task_notes(with_extra)


def test_new_task_widens_and_does_not_shift(fitted):
    _, _, st = fitted
    plain = Task(id="tsk_new_problem", type="feature", repo="acme/api")
    traits = Task(id="tsk_new_problem", type="feature", repo="acme/api",
                  features={"domain": "web", "input_items": "large"})
    mu0, _, _, s2_0 = _task_part(st, plain)
    mu1, _, _, s2_1 = _task_part(st, traits)
    assert mu1[0] == pytest.approx(mu0[0]) and s2_1[0] > s2_0[0]
    notes = st.task_notes(traits)
    assert "domain=web: 1 task, not used yet; the range is wider for it" in notes
    assert ("input_items=large: input_items splits the tasks as output does, not used yet; "
            "the range is wider for it") in notes
    admitted = Task(id="tsk_new_problem", type="feature", repo="acme/api", features={"output": "actions"})
    mu2, _, _, s2_2 = _task_part(st, admitted)
    assert s2_2[0] == pytest.approx(s2_0[0]) and mu2[0] != pytest.approx(mu0[0])  # a fitted node shifts


def test_covered_values_add_nothing(fitted):
    """A constant key's value and an alias value paired as in the fitted tasks: no node, no widening."""
    _, _, st = fitted

    def part(**feats):
        return _task_part(st, Task(id="tsk_new_problem", type="feature", repo="acme/api", features=feats))

    mu, _, _, s2 = part(output="structure")
    for same in (dict(lang="cpp"), dict(input_items="large"), dict(input_items="9000", lang="cpp")):
        mu_s, _, _, s2_s = part(output="structure", **same)
        assert mu_s[0] == pytest.approx(mu[0]) and s2_s[0] == pytest.approx(s2[0]), same
    for other in (dict(lang="python"), dict(input_items="small")):
        mu_o, _, _, s2_o = part(output="structure", **other)
        assert mu_o[0] == pytest.approx(mu[0]) and s2_o[0] > s2[0], other
    task = Task(id="tsk_new_problem", type="feature", repo="acme/api",
                features={"output": "structure", "lang": "cpp", "input_items": "small"})
    notes = st.task_notes(task)
    assert not any(n.startswith("lang=") for n in notes)
    assert any(n.startswith("input_items=small") for n in notes)


def test_known_task_inherits_its_recorded_values(fitted):
    _, _, st = fitted
    tid = _tid(0)
    bare = Task(id=tid, type=simdata.TYPES[0], repo=simdata.REPOS[0])
    full = Task(id=tid, type=simdata.TYPES[0], repo=simdata.REPOS[0], features={"output": "structure", "grid": "yes"})
    resolved, info = st.resolve_task(bare)
    assert info["inherited"] == {"output": "structure", "grid": "yes"}
    assert resolved.features == {"output": "structure", "grid": "yes"}
    assert _task_part(st, bare)[0][0] == pytest.approx(_task_part(st, full)[0][0])
    assert "features as recorded for this task: grid=yes, output=structure" in st.task_notes(bare)
    given = Task(id=tid, type=simdata.TYPES[0], repo=simdata.REPOS[0], features={"output": "actions"})
    assert st.resolve_task(given)[1]["inherited"] == {"grid": "yes"}  # a given value wins


def test_fit_summary_line(fitted, capsys):
    _, path, _ = fitted
    _print_summary(_summary(path))
    out = capsys.readouterr().out
    assert ("features: output 2 values (2+2 tasks), grid 2 values (2+2 tasks); dropped lang (constant), "
            "input_items (alias of output), domain (below min_tasks)") in out


def test_a_bad_declaration_fits_with_the_builtins(tmp_path):
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    for doc in _docs(40, seed=62):
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    (home / "config.toml").write_text('[features.size]\nvalues = ["a"]\n', encoding="utf-8")
    meta = json.loads((F.fit(home, no_prior=True, now=NOW) / "meta.json").read_text())
    assert "built-in" in meta["features"]["error"] and meta["features"]["declared"]["custom"] == []
    assert "config [features] not used" in features_line(meta["features"])


def test_posterior_knows_declared_keys(fitted, capsys):
    home, _, _ = fitted
    argv = ["posterior", "--home", str(home), "--type", "feature", "--repo", "acme/api", "--level", "feature",
            "--feature", "domain=web", "--feature", "colour=red", "--json"]
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    page = json.loads(out)
    assert page["task"]["features"]["domain"] == "web"  # the fit's declared key, not extra:domain
    assert "colour" in err and "domain" not in err  # only the undeclared key is "not modelled"


def test_undeclared_then_declared_reads_as_declared():
    fs = FeatureSet.from_config(DECLARED)
    assert fs.normalize({"extra:output": "Actions", "extra:horizon_s": "7200", "extra:colour": "red"}) == {
        "output": "actions", "horizon_s": "7200", "extra:colour": "red"}
    assert fs.normalize({"output": "structure", "extra:output": "actions"}) == {"output": "structure"}
