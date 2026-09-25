"""`loopmath prior import-shared`: strict check, organization group, merge, reader (lane 08)."""

from __future__ import annotations

import copy
import gzip
import itertools
import json

import pytest
from share_store import PLANTED, planted_docs, write_store

from loopmath import cli
from loopmath.ocp import conformance
from loopmath.share import import_
from loopmath.share.export import EXT_KEY, write_share
from loopmath.share.import_ import check_share


def _share(tmp_path, home, name="share.json.gz") -> tuple:
    out = tmp_path / name
    assert cli.main(["share", "--out", str(out), "--home", str(home), "--json"]) == 0
    return out, json.loads(gzip.decompress(out.read_bytes()))


def _import(ours, path, *extra) -> int:
    return cli.main(["prior", "import-shared", str(path), "--home", str(ours), *extra])


def test_round_trip_adds_runs_under_a_new_org_node(store, tmp_path, capsys):
    path, obj = _share(tmp_path, store)
    capsys.readouterr()
    ours = tmp_path / "ours"
    assert _import(ours, path, "--json") == 0
    result = json.loads(capsys.readouterr().out)
    org = "shared:" + obj["org_hash"]
    assert next(iter(result)) == "schema" and result["schema"] == "loopmath.prior.import-shared/1"
    assert result["org"] == org and result["added"] == 3 and result["total"] == 3
    assert result["path"] == str(ours / "priors" / f"shared-{obj['org_hash']}.json.gz")

    runs = list(import_.shared_runs(ours))
    assert len(runs) == 3
    for doc in runs:
        assert doc["run"]["task"]["org"] == org
        assert doc["run"]["ext"][EXT_KEY]["source"] == org
        assert set(doc["run"]["ext"][EXT_KEY]["outcome"]) == {"z", "q", "tier"}
        assert doc["ocp"] == "0.3" and doc["privacy"] == {"profile": "metadata_only"}
    stored = {d["run"]["id"]: d for d in obj["runs"]}
    for doc in runs:  # import stamps the share's time and the reader adds the org node; nothing else changes
        doc["run"]["task"].pop("org")
        doc["run"]["ext"][EXT_KEY].pop("source")
        assert doc["run"]["ext"][EXT_KEY].pop("shared_at") == obj["created_at"]
        assert doc == stored[doc["run"]["id"]]


def test_reimport_is_idempotent_and_later_shares_merge(tmp_path, capsys):
    home = write_store(tmp_path / "home", planted_docs()[:2])
    first, _ = _share(tmp_path, home, "first.json.gz")
    ours = tmp_path / "ours"
    assert _import(ours, first) == 0
    capsys.readouterr()
    assert _import(ours, first, "--json") == 0
    again = json.loads(capsys.readouterr().out)
    assert (again["added"], again["updated"], again["unchanged"], again["total"]) == (0, 0, 2, 2)

    write_store(home, planted_docs()[2:3])  # one more finished run in the same store
    second, _ = _share(tmp_path, home, "second.json.gz")
    capsys.readouterr()
    assert _import(ours, second, "--json") == 0
    merged = json.loads(capsys.readouterr().out)
    assert (merged["added"], merged["unchanged"], merged["total"]) == (1, 2, 3)
    assert len(list(import_.shared_runs(ours))) == 3
    assert len(list((ours / "priors").iterdir())) == 1  # one organization, one file


def _partial(obj, at, zs):
    """A share of the same store made at `at`, holding only the runs at the indexes in `zs`, each with that z."""
    part = copy.deepcopy(obj)
    part["created_at"] = at
    part["runs"] = []
    for i, z in zs.items():
        doc = copy.deepcopy(obj["runs"][i])
        doc["run"]["ext"][EXT_KEY]["outcome"]["z"] = z
        part["runs"].append(doc)
    return part


# 10:00, 11:00 and 12:00 UTC, written so that their text sorts the other way round
T10, T11, T12 = "2026-09-01T20:00:00+10:00", "2026-09-01T11:00:00Z", "2026-09-01T05:00:00-07:00"


@pytest.mark.parametrize("order", list(itertools.permutations("ABC")))
def test_the_later_share_wins_each_run_whatever_the_import_order(store, tmp_path, order):
    """A (10:00) has x and w, B (12:00) has y and w, C (11:00) has x only: x comes from C, y and w from B."""
    _, obj = _share(tmp_path, store)
    x, y, w = (d["run"]["id"] for d in obj["runs"])
    files = {"A": _partial(obj, T10, {0: 0.0, 2: 0.75}), "B": _partial(obj, T12, {1: 0.5, 2: 0.25}),
             "C": _partial(obj, T11, {0: 1.0})}
    ours = tmp_path / "ours"
    for name in order:
        assert check_share(files[name]) == []
        import_.import_share(files[name], ours)
    kept = {d["run"]["id"]: d["run"]["ext"][EXT_KEY] for d in import_.shared_runs(ours)}
    assert {k: (e["outcome"]["z"], e["shared_at"]) for k, e in kept.items()} == {
        x: (1.0, T11), y: (0.5, T12), w: (0.25, T12)}
    assert import_.read_share(import_.priors_path(ours, obj["org_hash"]))["created_at"] == T12


def test_an_older_share_counts_as_unchanged(store, tmp_path):
    _, obj = _share(tmp_path, store)
    older = _partial(obj, "2026-09-01T09:00:00-07:00", {0: 0.0, 1: 0.0, 2: 0.0})
    ours = tmp_path / "ours"
    import_.import_share(obj, ours)
    result = import_.import_share(older, ours)
    assert (result["added"], result["updated"], result["unchanged"], result["total"]) == (0, 0, 3, 3)
    newer = _partial(obj, "2026-12-01T09:00:00-08:00", {0: 0.5, 1: obj["runs"][1]["run"]["ext"][EXT_KEY]["outcome"]["z"]})
    result = import_.import_share(newer, ours)
    assert (result["added"], result["updated"], result["unchanged"], result["total"]) == (0, 1, 1, 3)


def _allocated(obj) -> dict:
    """The one allocated cost in the planted share (the heuristic implement attempt)."""
    (cost,) = [a["cost"] for d in obj["runs"] for a in d["attempts"] if a["cost"]["basis"] == "allocated"]
    return cost


LOGMATCH = "dev.loopmath.logmatch"
SPLIT = "dev.loopmath.model_tokens"


def _bad(obj, tmp_path, name="bad.json.gz"):
    path = tmp_path / name
    write_share(obj, path)
    return path


@pytest.mark.parametrize("mutate, where", [
    (lambda o: o.update(schema="loopmath.share/2"), "schema"),
    (lambda o: o.update(owner="Alice at Acme"), "share.<key>"),
    (lambda o: o["runs"][0]["attempts"][0].update(session=PLANTED["claude session"]), ".attempts[0].<key>"),
    (lambda o: o["runs"][0]["run"]["task"].update(title=PLANTED["task title"]), ".run.task.<key>"),
    (lambda o: o["runs"][0]["run"]["task"].update(subtype="/Users/alice/acme-secret"), ".run.task.subtype"),
    (lambda o: o["runs"][0]["run"]["task"].update(subtype="Users/alice/acme-secret"), ".run.task.subtype"),
    (lambda o: o["runs"][0]["run"]["acceptance_rule"].update(definition=PLANTED["rule definition"]),
     ".acceptance_rule.definition"),
    (lambda o: o["runs"][0]["run"]["acceptance_rule"].update(name=PLANTED["rule name"]), ".acceptance_rule.name"),
    (lambda o: o["runs"][0]["run"]["task"].update(repo=PLANTED["repo"]), ".run.task.repo"),
    (lambda o: o["runs"][0]["nodes"][0].update(vertex="fix the acme login"), ".nodes[0].vertex"),
    (lambda o: o["runs"][0]["attempts"][0].update(effort=PLANTED["claude session"]), ".attempts[0].effort"),
    (lambda o: o["runs"][0]["run"]["configuration"]["workflow"].update(id=PLANTED["base commit"]),
     ".configuration.workflow"),
    (lambda o: o["runs"][0]["run"]["ext"][EXT_KEY]["outcome"].update(q=2), ".outcome.q"),
    (lambda o: o["runs"].append(copy.deepcopy(o["runs"][0])), "same run id"),
    # The only cost ext is the log match tier, on an allocated cost
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(session=PLANTED["claude session"]),
     f".cost.ext.{LOGMATCH}.<key>"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(clip={"from": "2026-09-20T10:00:00-07:00"}),
     f".cost.ext.{LOGMATCH}.<key>"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(tier="verified"), f".cost.ext.{LOGMATCH}.tier"),
    (lambda o: _allocated(o)["ext"].update({"com.acme.cost": PLANTED["ext value"]}), ".cost.ext.<key>"),
    (lambda o: _allocated(o).pop("ext"), ".cost.ext"),
    (lambda o: o["runs"][0]["attempts"][0]["cost"].update(ext={LOGMATCH: {"tier": "heuristic"}}), ".cost.ext"),
    (lambda o: o["runs"][0]["attempts"][0]["cost"].update(basis="expected"), ".cost.basis"),
    (lambda o: _allocated(o)["ext"].pop(LOGMATCH), f".cost.ext.{LOGMATCH}"),
    (lambda o: o["runs"][0]["attempts"][0]["cost"].update(ext={}), ".cost.ext"),
    # A split is {model id: the four OCP token counts}, nothing more
    (lambda o: _allocated(o)["ext"][SPLIT]["gpt-6-astra"].update(session=PLANTED["codex session"]),
     f".cost.ext.{SPLIT}.gpt-6-astra.<key>"),
    (lambda o: _allocated(o)["ext"][SPLIT]["gpt-6-astra"].update(reasoning_tokens=5),
     f".cost.ext.{SPLIT}.gpt-6-astra.<key>"),
    (lambda o: _allocated(o)["ext"][SPLIT]["gpt-6-astra"].pop("output_tokens"),
     f".cost.ext.{SPLIT}.gpt-6-astra.output_tokens"),
    (lambda o: _allocated(o)["ext"][SPLIT]["gpt-6-astra"].update(input_tokens=-1),
     f".cost.ext.{SPLIT}.gpt-6-astra.input_tokens"),
    (lambda o: _allocated(o)["ext"][SPLIT].update({PLANTED["model label"]: _allocated(o)["ext"][SPLIT]["gpt-6-astra"]}),
     f".cost.ext.{SPLIT}.<key>"),
    (lambda o: _allocated(o)["ext"].update({SPLIT: [["gpt-6-astra", 1]]}), f".cost.ext.{SPLIT}"),
    # A shared session adds a bare `shared_session: true`, and only then may the tier be verified
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session=1), f".cost.ext.{LOGMATCH}.shared_session"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session="true"), f".cost.ext.{LOGMATCH}.shared_session"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session=False), f".cost.ext.{LOGMATCH}.shared_session"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session={"session": PLANTED["claude session"]}),
     f".cost.ext.{LOGMATCH}.shared_session"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session=True, attempts=["att_1", "att_2"]),
     f".cost.ext.{LOGMATCH}.<key>"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session=True, tier="reported"),
     f".cost.ext.{LOGMATCH}.tier"),
    (lambda o: _allocated(o)["ext"][LOGMATCH].update(shared_session=True, tier=None), f".cost.ext.{LOGMATCH}.tier"),
    (lambda o: o["runs"][0]["attempts"][0]["cost"].update(ext={LOGMATCH: {"tier": "verified", "shared_session": True}}),
     f".cost.ext.{LOGMATCH}"),
])
def test_import_rejects_files_that_carry_more_than_the_format(store, tmp_path, capsys, mutate, where):
    _, obj = _share(tmp_path, store)
    mutate(obj)
    ours = tmp_path / "ours"
    assert _import(ours, _bad(obj, tmp_path)) == 1
    err = capsys.readouterr().err
    assert where in err
    for value in (PLANTED["claude session"], PLANTED["task title"], "alice", "acme"):
        assert value.lower() not in err.lower(), "problems name the place, never the value"
    assert not (ours / "priors").exists()


def test_allocated_cost_survives_the_round_trip(store, tmp_path, capsys):
    path, obj = _share(tmp_path, store)
    ext = _allocated(obj)["ext"]
    assert ext[LOGMATCH] == {"tier": "heuristic"} and set(ext) == {LOGMATCH, SPLIT}
    ours = tmp_path / "ours"
    assert _import(ours, path) == 0
    costs = [a["cost"] for d in import_.shared_runs(ours) for a in d["attempts"]]
    assert [c["ext"] for c in costs if c["basis"] == "allocated"] == [ext]
    assert all(set(c.get("ext", {})) <= {SPLIT} for c in costs if c["basis"] == "measured")


@pytest.mark.parametrize("tier", ["heuristic", "verified"])
def test_shared_session_survives_the_round_trip(tmp_path, capsys, tier):
    """Attempts that split one session share their tier and `shared_session: true`, and import takes exactly
    that; which session and which attempts stay home. A verified split needs lane 1's E171 change."""
    docs = planted_docs()
    (cost,) = [a["cost"] for a in docs[0]["attempts"] if a["cost"]["basis"] == "allocated"]
    cost["ext"][LOGMATCH].update(tier=tier, shared_session={"session": PLANTED["codex session"],
                                                            "attempts": ["att_impl_1", "att_impl_2"]})
    path, obj = _share(tmp_path, write_store(tmp_path / "theirs", docs))
    assert _allocated(obj)["ext"][LOGMATCH] == {"tier": tier, "shared_session": True}
    assert check_share(obj) == []
    problems = import_.ocp_problems(obj)
    if tier == "verified" and problems and all(p.endswith("OCP rule E171") for p in problems):
        pytest.skip("lane 01's D88 E171 change (allocated with a shared session) is not on this tree")
    assert problems == []
    ours = tmp_path / "ours"
    assert _import(ours, path) == 0
    (stored,) = [a["cost"]["ext"][LOGMATCH] for d in import_.shared_runs(ours) for a in d["attempts"]
                 if a["cost"]["basis"] == "allocated"]
    assert stored == {"tier": tier, "shared_session": True}


def test_model_token_splits_survive_the_round_trip(store, tmp_path, capsys):
    """A split with an unknown part, a hashed private label, and `{}` (mixed, no split) all import as shared."""
    path, obj = _share(tmp_path, store)
    shared = [a["cost"]["ext"][SPLIT] for d in obj["runs"] for a in d["attempts"] if SPLIT in a["cost"].get("ext", {})]
    assert {} in shared and any("unknown" in s for s in shared)
    assert any(k.startswith("h_") for s in shared for k in s)
    ours = tmp_path / "ours"
    assert _import(ours, path) == 0
    stored = [a["cost"]["ext"][SPLIT] for d in import_.shared_runs(ours) for a in d["attempts"]
              if SPLIT in a["cost"].get("ext", {})]
    assert sorted(map(json.dumps, stored)) == sorted(map(json.dumps, shared))


def test_allocated_cost_round_trip_passes_the_ocp_checker(store, tmp_path, capsys):
    """Review of 0990dd5: an allocated share goes through lane 1's checker on import and after it."""
    path, obj = _share(tmp_path, store)
    assert import_.ocp_problems(obj) == []
    ours = tmp_path / "ours"
    assert _import(ours, path) == 0
    docs = list(import_.shared_runs(ours))
    assert any(a["cost"]["basis"] == "allocated" for d in docs for a in d["attempts"])
    for doc in docs:
        assert [f for f in conformance.validate_doc(doc) if f.level == "error"] == []
    del _allocated(obj)["ext"]  # without the tier, E171 would refuse the cost; the shape check says so first
    assert _import(tmp_path / "other", _bad(obj, tmp_path)) == 1


def test_import_rejects_unreadable_and_missing_files(tmp_path, capsys):
    ours = tmp_path / "ours"
    junk = tmp_path / "junk.json.gz"
    junk.write_bytes(b"\x1f\x8b not really gzip")
    assert _import(ours, junk) == 1
    text = tmp_path / "list.json"
    text.write_text("[1, 2]")
    assert _import(ours, text) == 1
    assert _import(ours, tmp_path / "nope.json.gz") == 2
    assert "no such file" in capsys.readouterr().err


def test_import_refuses_files_over_the_size_limit(store, tmp_path, capsys, monkeypatch):
    path, obj = _share(tmp_path, store)
    plain = tmp_path / "share.json"
    plain.write_text(json.dumps(obj))
    limit = path.stat().st_size + 10  # the gzip file fits on disk; neither its content nor the plain file does
    assert limit < plain.stat().st_size
    monkeypatch.setattr(import_, "MAX_BYTES", limit)
    capsys.readouterr()
    ours = tmp_path / "ours"
    assert _import(ours, plain) == 1
    assert _import(ours, path) == 1
    assert capsys.readouterr().err.count(f"larger than {limit} bytes") == 2
    assert not (ours / "priors").exists()


def test_import_runs_the_ocp_checker(store, tmp_path, capsys):
    """A file whose configuration id is not the hash of its content is refused, by place and rule only."""
    _, obj = _share(tmp_path, store)
    doc = next(d for d in obj["runs"] if "pieces" in d["run"]["configuration"]["workflow"])
    doc["run"]["configuration"]["id"] = "cfg_000000000000"
    capsys.readouterr()
    assert _import(tmp_path / "ours", _bad(obj, tmp_path)) == 1
    err = capsys.readouterr().err
    assert "OCP rule E190" in err and "cfg_000000000000" not in err


def test_plain_json_is_accepted(store, tmp_path, capsys):
    _, obj = _share(tmp_path, store)
    plain = tmp_path / "share.json"
    plain.write_text(json.dumps(obj))
    assert _import(tmp_path / "ours", plain) == 0
    assert "imported 3 runs as organization shared:" in capsys.readouterr().out


def test_reader_skips_a_damaged_stored_file(store, tmp_path, capsys):
    path, obj = _share(tmp_path, store)
    ours = tmp_path / "ours"
    assert _import(ours, path) == 0
    (ours / "priors" / "shared-0000000000000000.json.gz").write_bytes(b"garbage")
    assert len(list(import_.shared_runs(ours))) == 3
    assert "skipping" in capsys.readouterr().err
    # and a damaged file for the same organization blocks a merge instead of being overwritten
    import_.priors_path(ours, obj["org_hash"]).write_bytes(gzip.compress(b'{"schema": "loopmath.share/1"}'))
    assert _import(ours, path) == 1
    assert "move it away" in capsys.readouterr().err


def test_fit_counts_an_import_under_the_name_import_shared_prints(store, tmp_path, capsys):
    """22X: share, import-shared, fit; the source is `shared:<org_hash>` and `--without` takes that name."""
    path, obj = _share(tmp_path, store)
    ours = tmp_path / "ours"
    capsys.readouterr()
    assert _import(ours, path) == 0
    org = "shared:" + obj["org_hash"]
    assert f"imported 3 runs as organization {org}:" in capsys.readouterr().out
    assert cli.main(["fit", "--home", str(ours), "--json"]) == 0
    fitted = json.loads(capsys.readouterr().out)
    assert fitted["runs_by_source"][org] == 3 and not any(s.startswith("shared:shared:") for s in fitted["runs_by_source"])
    assert cli.main(["fit", "--home", str(ours), "--without", org, "--json"]) == 0
    out, err = capsys.readouterr()
    assert "unknown source" not in err
    without = json.loads(out)
    assert without["options"]["without"] == [org] and org not in without["runs_by_source"]
    assert without["dropped"][f"without {org}"] == 3
    meta = json.loads((ours / "fits" / "latest" / "meta.json").read_text())
    assert meta["options"]["without"] == [org] and org not in meta["runs_by_source"]
