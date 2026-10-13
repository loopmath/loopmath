"""A shared run's outcome reaches the receiver (0.2.7).

`share` writes each run's outcome as `ext["dev.loopmath.share"]["outcome"] = {z, q, tier}` and its scores as a
list. Before 0.2.7 the receiver read only an `evidence` object or z and q at the top of that ext, so every
imported run had an unknown outcome and the receiver's success head never saw a shared run. The receiver now
reads the shape the export writes and still reads the old ones. A shared run that carries no q takes the
receiver's configured q for its tier, as a local run does.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime

import pytest

from loopmath import cli
from loopmath.belief import fit as F
from loopmath.belief.design import rule_from_doc
from loopmath.belief.outcome import TIER_Q, outcome_evidence
from loopmath.share import import_
from loopmath.share.export import run_rule, store_docs

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")


def _run(argv, capsys):
    capsys.readouterr()
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def _receiver(store, tmp_path, capsys):
    out = tmp_path / "share.json"
    assert _run(["share", "--out", str(out), "--home", str(store)], capsys)[0] == 0
    ours = tmp_path / "ours"
    code, _, err = _run(["prior", "import-shared", str(out), "--home", str(ours)], capsys)
    assert code == 0, err
    return ours, json.loads(out.read_text())


def _shared(z=1.0, q=0.9, tier="verified", **extra):
    return {"run": {"id": "run_x", "ext": {"dev.loopmath.share": {"outcome": {"z": z, "q": q, "tier": tier}, **extra}}}}


def _ev(doc, q_by_tier=None):
    return outcome_evidence(doc, rule_from_doc(doc), now=NOW, q_by_tier=q_by_tier)


def test_the_receiver_reads_each_shared_outcome(store, tmp_path, capsys):
    ours, share = _receiver(store, tmp_path, capsys)
    carried = {r["run"]["id"]: r["run"]["ext"]["dev.loopmath.share"]["outcome"] for r in share["runs"]}
    seen = 0
    for doc in import_.shared_runs(ours):
        ev = _ev(doc)
        want = carried[doc["run"]["id"]]
        assert (ev.z, ev.q, ev.tier) == (want["z"], want["q"], want["tier"])
        seen += 1
    assert seen == len(carried) > 0


def test_the_receivers_fit_has_shared_success_rows(store, tmp_path, capsys):
    known = sum(outcome_evidence(d, run_rule(d), now=NOW).z is not None for d in store_docs(store))
    assert known > 0
    ours, _ = _receiver(store, tmp_path, capsys)
    path = F.fit(ours, docs=list(import_.shared_runs(ours)), no_prior=True, now=NOW, bundle_dir=ours / "no-bundle")
    meta = json.loads((path / "meta.json").read_text())
    rows = (meta["heads"].get("success") or {}).get("rows_by_source") or {}
    assert sum(v for k, v in rows.items() if k.startswith("shared:")) == known


@pytest.mark.parametrize("ext", [
    {"evidence": {"z": 1.0, "q": 0.9, "tier": "verified"}},
    {"z": 1.0, "q": 0.9, "tier": "verified"},
])
def test_the_old_shapes_still_read(ext):
    ev = _ev({"run": {"id": "run_x", "ext": {"dev.loopmath.share": ext}}})
    assert (ev.z, ev.q, ev.tier) == (1.0, 0.9, "verified")


def test_scores_read_as_a_list_or_a_dict():
    listed = _shared(scores=[{"name": "perf", "value": 2.5, "better": "higher"}, {"name": "bad", "value": "x"}])
    assert _ev(listed).scores == {"perf": 2.5}
    keyed = _shared(scores={"perf": 2.5})
    assert _ev(keyed).scores == {"perf": 2.5}


def test_a_carried_q_is_kept_whatever_the_receivers_config():
    doc = _shared(q=0.9, tier="reported")
    assert _ev(doc).q == 0.9
    assert _ev(doc, {"reported": 0.6}).q == 0.9


def test_no_carried_q_takes_the_receivers_configured_q():
    doc = _shared(q=None, tier="reported")
    assert _ev(doc).q == TIER_Q["reported"]
    assert _ev(doc, {**TIER_Q, "reported": 0.6}).q == 0.6
    assert _ev(doc, {**TIER_Q, "verified": 0.6}).q == TIER_Q["reported"]


def test_a_local_run_is_unchanged_by_the_shared_reader():
    local = {"run": {"id": "run_l", "ended_at": "2026-09-20T11:00:00-07:00", "signals": [
        {"id": "s", "kind": "verdict", "name": "tests", "value": "pass", "observed_at": "2026-09-20T11:00:00-07:00",
         "tier": "reported"}]}}
    assert _ev(copy.deepcopy(local), {**TIER_Q, "reported": 0.6}).q == 0.6
    assert _ev(local).z == 1.0
