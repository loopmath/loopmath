"""Dollars no float holds (an attempt's `cost.usd` an integer `float()` cannot convert): a run to be finished with
them is refused by name before anything is written, by `Store.import_run`, `Store.finish`, `run finish` and
`run import --finish` alike, so no run file, index row, pending signal or receipt changes. `run import` without
`--finish` stores such a run open, as before; the rest of a batch goes on as before."""

from __future__ import annotations

import hashlib
import json

import pytest
from store_helpers import cli, finished_doc

from loopmath.store import Store, fitjob
from loopmath.store import runs as R
from loopmath.store.home import DollarsPastFloat
from loopmath.types import Signal

BIG = 10 ** 400
WHERE = "attempts[0].cost.usd"
BEFORE = {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
          "cost": {"usd": {"mean": 1.0, "lo": 0.5, "hi": 1.5, "level": 0.8},
                   "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "ell": {"usd": {"mean": 1.2, "lo": 0.6, "hi": 1.8, "level": 0.8},
                  "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "rounds": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8}, "per_piece": {}, "support": 3}


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    """The refits started, one entry each."""
    started: list[str] = []

    def spawn(home, **kw):
        started.append(str(home))
        return {"started": True, "pid": 0}

    monkeypatch.setattr(fitjob, "spawn_fit", spawn)
    return started


def _refusal(run: str, where: str = WHERE) -> str:
    return f"run {run}: {where} is past what a float holds, so the run cannot be finished; nothing was written"


def _big(run: str) -> dict:
    doc = finished_doc(run, usd=0.1)
    doc["attempts"][0]["cost"]["usd"] = BIG
    return doc


def _write(path, doc) -> str:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _tree(home) -> dict[str, str]:
    return {str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(home.rglob("*")) if p.is_file()} if home.exists() else {}


def _rows(home, run: str) -> list[dict]:
    store = Store(home)
    if not store.index_path.exists():
        return []
    return [json.loads(line) for line in store.index_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("run") == run]


def _open_with_signal_and_receipt(home, run: str) -> Store:
    """The run stored open, with a pending signal and the receipt `run start --rec` writes."""
    store = Store(home)
    store.import_run(_big(run), finished=False)
    assert store.add_signal(run, Signal(id="sig_pending1", run=run, kind="verdict", name="tests", value="pass",
                                        tier="verified", source={"kind": "ci"},
                                        observed_at="2026-09-20T11:00:00-07:00")) == "pending"
    store.write_receipt({"id": f"rct_{run}", "rec": None, "run": run, "fit": "fit_1", "before": BEFORE})
    return store


# ------------------------------------------------------------------ the check


@pytest.mark.parametrize(("attempts", "where"), [
    ([], None), ([{"cost": {"usd": 1e308}}], None), ([{"cost": {"usd": 10 ** 308}}], None),
    ([{"cost": {"usd": True}}], None), ([{"cost": "x"}, {"cost": {}}], None),
    ([{"cost": {"usd": 0.1}}, {"cost": {"usd": 10 ** 309}}, {"cost": {"usd": BIG}}], "attempts[1].cost.usd"),
    (["not an attempt", {"cost": {"usd": -BIG}}], "attempts[1].cost.usd")],
    ids=["none", "float", "int-a-float-holds", "bool", "no-dollars", "first-named", "index-counts-every-item"])
def test_the_check_names_the_first_attempt_whose_dollars_no_float_holds(attempts, where):
    assert R.dollars_past_float({"attempts": attempts}) == where


# ------------------------------------------------------------------ the store


@pytest.mark.parametrize("how", ["finished", "store-ext-finished"])
def test_store_import_run_refuses_a_run_to_be_finished_and_writes_nothing(how, tmp_path):
    """Finished by the argument, or by the document's own store ext with `finished=False`: either way finished."""
    home, doc = tmp_path / "lm", _big("run-big")
    if how == "store-ext-finished":
        R.set_store_ext(doc, state=R.FINISHED)
    with pytest.raises(DollarsPastFloat) as caught:
        Store(home).import_run(doc, finished=how == "finished")
    assert (str(caught.value), caught.value.run, caught.value.path) == (_refusal("run-big"), "run-big", WHERE)
    assert _tree(home) == {}


def test_store_import_run_stores_it_open_as_before(tmp_path):
    home = tmp_path / "lm"
    Store(home).import_run(_big("run-big"), finished=False)
    assert Store(home).run_doc("run-big")["attempts"][0]["cost"]["usd"] == BIG
    assert [r["state"] for r in _rows(home, "run-big")] == ["open"]


def test_store_finish_refuses_and_leaves_the_run_open(tmp_path):
    home = tmp_path / "lm"
    store = _open_with_signal_and_receipt(home, "run-big")
    before = _tree(home)
    with pytest.raises(DollarsPastFloat, match=_refusal("run-big").replace("[", r"\[").replace("]", r"\]")):
        store.finish("run-big", store.run_doc("run-big"))
    assert _tree(home) == before and not R.is_finished(store.run_doc("run-big"))


@pytest.mark.parametrize("receipt", [False, True], ids=["no-receipt", "receipt"])
@pytest.mark.parametrize("json_", [False, True], ids=["text", "json"])
def test_run_finish_refuses_by_name_and_changes_nothing(receipt, json_, capsys, tmp_path, fits):
    """With the receipt `run start --rec` wrote, the receipt reads the attempts' dollars too: refused first."""
    home = tmp_path / "lm"
    store = _open_with_signal_and_receipt(home, "run-big") if receipt else Store(home)
    if not receipt:
        store.import_run(_big("run-big"), finished=False)
    before = _tree(home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", "run-big", *(["--json"] if json_ else []))
    assert code == 1 and "Traceback" not in err
    if json_:
        assert out == {"schema": "loopmath.run.finish/1", "ok": False, "error": _refusal("run-big"), "exit": 1}
    else:
        assert (out, err) == ("", f"error: {_refusal('run-big')}\n")
    assert _tree(home) == before and fits == []


# ------------------------------------------------------------------ run import


@pytest.mark.parametrize("how", ["--finish", "store-ext-finished"])
def test_run_import_of_one_file_refuses_it_before_it_is_stored_open(how, capsys, tmp_path, fits):
    home, doc = tmp_path / "lm", _big("run-big")
    if how == "store-ext-finished":
        R.set_store_ext(doc, state=R.FINISHED)
    path = _write(tmp_path / "big.ocp.json", doc)
    flags = ["--finish"] if how == "--finish" else []
    code, out, err = cli(capsys, home, "run", "import", path, *flags)
    assert (code, out, err) == (1, "", f"error: {_refusal('run-big')}\n")
    code, out, err = cli(capsys, home, "run", "import", path, *flags, "--json")
    assert code == 1 and out == {"schema": "loopmath.run.import/1", "ok": False, "error": _refusal("run-big"),
                                 "exit": 1}
    assert not Store(home).exists("run-big") and _rows(home, "run-big") == [] and fits == []


def test_run_import_without_finish_stores_it_open_as_before(capsys, tmp_path, fits):
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "run", "import", _write(tmp_path / "big.ocp.json", _big("run-big")), "--json")
    assert code == 0 and (out["run"], out["state"], out["import"]) == ("run-big", "open", "new")
    assert [r["state"] for r in _rows(home, "run-big")] == ["open"] and fits == []


def test_run_import_finish_of_a_run_stored_open_changes_nothing_of_it(capsys, tmp_path, fits):
    """The run, its index rows, its pending signal and its receipt stay as they were."""
    home = tmp_path / "lm"
    _open_with_signal_and_receipt(home, "run-big")
    before = _tree(home)
    code, out, err = cli(capsys, home, "run", "import", _write(tmp_path / "big.ocp.json", _big("run-big")),
                         "--finish")
    assert (code, err) == (1, f"error: {_refusal('run-big')}\n")
    assert _tree(home) == before and fits == []


@pytest.mark.parametrize("given", ["folder", "files"])
def test_in_a_batch_the_run_is_refused_by_run_and_path_and_the_rest_finished(given, capsys, tmp_path, fits):
    home, src = tmp_path / "lm", tmp_path / "ocp"
    src.mkdir()
    files = [_write(src / "a.ocp.json", finished_doc("run-real", usd=0.1)), _write(src / "b.ocp.json", _big("run-big"))]
    args = [str(src)] if given == "folder" else files
    code, out, err = cli(capsys, home, "run", "import", *args, "--finish", "--json")
    assert code == 1 and out["runs"] == ["run-real"] and (out["failed"], out["refused"], out["finished"]) == (1, 1, 1)
    [failed] = [f for f in out["files"] if not f["ok"]]
    assert failed == {"file": files[1], "ok": False, "error": _refusal("run-big"), "exit": 1,
                      "refused": {"run": "run-big", "past_float": WHERE}}
    assert Store(home).exists("run-real") and R.is_finished(Store(home).run_doc("run-real"))
    assert not Store(home).exists("run-big") and _rows(home, "run-big") == []
    assert len(fits) == 1 and out["fit"]["started"] is True


@pytest.mark.parametrize("form", ["text", "brief"])
def test_a_batch_names_the_refused_file_in_text_and_brief(form, capsys, tmp_path, fits):
    home, src = tmp_path / "lm", tmp_path / "ocp"
    src.mkdir()
    files = [_write(src / "a.ocp.json", finished_doc("run-real", usd=0.1)), _write(src / "b.ocp.json", _big("run-big"))]
    code, out, err = cli(capsys, home, "run", "import", *files, "--finish",
                         *(["--json", "--brief"] if form == "brief" else []))
    assert code == 1
    if form == "brief":
        brief = out
        assert (brief["imported"], brief["failed"], brief["refused"], brief["finished"]) == (1, 1, 1, 1)
        assert brief["failures"] == [{"file": files[1], "error": _refusal("run-big")}]
    else:
        assert f"FAIL  {files[1]}: {_refusal('run-big')}" in out.splitlines()
    assert not Store(home).exists("run-big") and len(fits) == 1


def test_a_batch_with_every_file_refused_starts_no_refit(capsys, tmp_path, fits):
    home, src = tmp_path / "lm", tmp_path / "ocp"
    src.mkdir()
    _write(src / "a.ocp.json", _big("run-big1"))
    _write(src / "b.ocp.json", _big("run-big2"))
    code, out, err = cli(capsys, home, "run", "import", str(src), "--finish", "--json")
    assert code == 1 and out["runs"] == [] and (out["failed"], out["refused"], out["finished"]) == (2, 2, 0)
    assert [f["refused"] for f in out["files"]] == [{"run": "run-big1", "past_float": WHERE},
                                                    {"run": "run-big2", "past_float": WHERE}]
    assert out["fit"]["started"] is False and fits == []
    assert not Store(home).exists("run-big1") and not Store(home).exists("run-big2")
