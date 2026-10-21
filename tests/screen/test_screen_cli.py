"""`loopmath screen`: a fold-over screen of atomic harness edits, recorded with `run record --screen`.

`start` takes a base harness tree, an edits folder (one subfolder per edit, an overlay on the base) and a tasks
file, and plans one fold-over pair per task: arm A runs the edits its row turns on, arm B the others. `next`
prints the next pair, with both arms' trees built in the store. `run record --screen SCR --arm A|B` records a
run into its pair. `status` ranks the edits on success and cost, names the predicted-best subset and its
predicted gain, and prices the next pair. `apply --best --out DIR` writes the base plus the best edits.
Synthetic stores, faked log settling with a per-run cost.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.store import finish as finish_mod
from loopmath.store.home import Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "store"))
from store_helpers import fake_settle  # noqa: E402

SOLO = ["--workflow", "solo", "--set", "implement=claude-code:claude-opus-5:high", "--source", "designed"]
SCREEN_EXT = "dev.loopmath.screen"


@pytest.fixture
def cost(monkeypatch):
    """The dollars the next recorded run settles at; a test sets `cost["usd"]` before each `run record`."""
    holder = {"usd": 0.5}
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle(usd=holder["usd"]))
    return holder


@pytest.fixture
def home(tmp_path, monkeypatch, cost):
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "logs"))
    (tmp_path / "wt").mkdir()
    h.mkdir()
    return h


def lm(capsys, *argv) -> tuple[int, dict | str, str]:
    capsys.readouterr()
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if "--json" in argv and out.strip() else out), err


def tasks_file(tmp_path: Path, n: int) -> Path:
    p = tmp_path / "tasks.jsonl"
    p.write_text("".join(json.dumps({"type": "bug_fix", "repo": "acme/app", "title": f"fix {i}",
                                     "base_commit": f"{i:07d}"}) + "\n" for i in range(n)))
    return p


def harness(tmp_path: Path, n_edits: int) -> tuple[Path, Path]:
    """A base tree with a CLAUDE.md, and `n_edits` edits that each append one line to it."""
    base, edits = tmp_path / "base", tmp_path / "edits"
    base.mkdir()
    (base / "CLAUDE.md").write_text("# Project rules\n")
    for j in range(n_edits):
        e = edits / f"e{j + 1:02d}-line"
        e.mkdir(parents=True)
        (e / "CLAUDE.md.append").write_text(f"- rule {j + 1}\n")
    return base, edits


def start(capsys, tmp_path, n_edits=3, n_tasks=4, *extra):
    base, edits = harness(tmp_path, n_edits)
    code, out, err = lm(capsys, "screen", "start", "--base", str(base), "--edits", str(edits), "--tasks",
                        str(tasks_file(tmp_path, n_tasks)), *extra, "--json")
    assert code == 0, err
    return out


def record(capsys, home, scr, arm, verdict="pass", *extra):
    code, out, err = lm(capsys, "run", "record", "--screen", scr, "--arm", arm, *SOLO,
                        "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--verified", f"tests={verdict}",
                        "--no-commits", "--no-fit", *extra, "--json")
    assert code == 0, err
    return out


# ---------------------------------------------------------------- the command
def test_the_command_is_registered_with_its_verbs(capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(["screen", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for verb in ("start", "next", "status", "apply"):
        assert verb in out
    assert "experimental" in out.lower()


# ---------------------------------------------------------------- start and next
def test_start_plans_mirror_pairs_and_builds_both_arms(home, tmp_path, capsys):
    out = start(capsys, tmp_path, 3, 4)
    assert out["schema"] == "loopmath.screen/1"
    scr = out["screen"]
    assert scr.startswith("scr_") and (home / "screens" / f"{scr}.json").is_file()
    assert [e["name"] for e in out["edits"]] == ["e01-line", "e02-line", "e03-line"]
    assert out["pairs"] == 4 and out["design"]["orthogonal"] is True
    nxt = out["next"]
    assert nxt["pair"] == "p01" and sorted(nxt["order"]) == ["A", "B"]
    on_a, on_b = nxt["arms"]["A"]["on"], nxt["arms"]["B"]["on"]
    assert sorted(on_a + on_b) == ["e01-line", "e02-line", "e03-line"] and not set(on_a) & set(on_b)
    for arm, on in (("A", on_a), ("B", on_b)):
        tree = Path(nxt["arms"][arm]["tree"])
        text = (tree / "CLAUDE.md").read_text()
        assert text.startswith("# Project rules\n")
        for j in (1, 2, 3):
            assert (f"- rule {j}\n" in text) == (f"e{j:02d}-line" in on)
        assert not any(p.name.endswith(".append") for p in tree.rglob("*"))
        assert nxt["arms"][arm]["harness"].startswith("sha256:")
    joined = "\n".join(nxt["commands"])
    assert f"--screen {scr} --arm A" in joined and f"--screen {scr} --arm B" in joined
    code, again, err = lm(capsys, "screen", "next", scr, "--json")
    assert code == 0, err
    assert again["next"]["pair"] == "p01" and again["next"]["arms"]["A"]["on"] == on_a


def test_every_edit_is_on_in_arm_a_for_half_the_pairs(home, tmp_path, capsys):
    out = start(capsys, tmp_path, 5, 8)
    rec = json.loads((home / "screens" / f"{out['screen']}.json").read_text())
    rows = np.array([p["row"] for p in rec["pairs"]])
    assert rows.shape == (8, 5) and (rows.sum(axis=0) == 0).all()
    assert (rows.T @ rows == 8 * np.eye(5)).all()


def test_start_refuses_edits_that_write_the_same_file_and_an_empty_edits_folder(home, tmp_path, capsys):
    base, edits = harness(tmp_path, 2)
    for name in ("e01-line", "e02-line"):
        (edits / name / "hooks.json").write_text("{}\n")
    code, out, err = lm(capsys, "screen", "start", "--base", str(base), "--edits", str(edits), "--tasks",
                        str(tasks_file(tmp_path, 4)), "--json")
    assert code == 1 and "hooks.json" in out["error"] and "e01-line" in out["error"]
    (tmp_path / "none").mkdir()
    code, out, err = lm(capsys, "screen", "start", "--base", str(base), "--edits", str(tmp_path / "none"),
                        "--tasks", str(tasks_file(tmp_path, 4)), "--json")
    assert code == 1 and "no edits" in out["error"]


def test_start_warns_when_there_are_fewer_pairs_than_edits_plus_one(home, tmp_path, capsys):
    out = start(capsys, tmp_path, 6, 4)
    assert out["design"]["orthogonal"] is False
    assert any("prior" in w for w in out["warnings"])


# ---------------------------------------------------------------- recording
def test_run_record_tags_the_run_with_its_pair_and_arm_and_the_pair_shares_a_task_and_slate(home, tmp_path, capsys):
    out = start(capsys, tmp_path, 3, 4)
    scr = out["screen"]
    a = record(capsys, home, scr, "A")
    b = record(capsys, home, scr, "B")
    st = Store(home)
    da, db = st.run_doc(a["run"]), st.run_doc(b["run"])
    tag_a = da["run"]["ext"][SCREEN_EXT]
    assert tag_a["screen"] == scr and tag_a["arm"] == "A" and tag_a["pair"] == "p01"
    assert tag_a["on"] == out["next"]["arms"]["A"]["on"]
    assert da["run"]["ext"]["dev.loopmath.pair"] == {"pair": "p01", "arm": "A", "experiment": scr}
    assert da["run"]["task"]["id"] == db["run"]["task"]["id"]
    assert da["run"]["slate"]["id"] == db["run"]["slate"]["id"]
    rec = json.loads((home / "screens" / f"{scr}.json").read_text())
    assert rec["pairs"][0]["runs"] == {"A": a["run"], "B": b["run"]}
    code, nxt, err = lm(capsys, "screen", "next", scr, "--json")
    assert nxt["next"]["pair"] == "p02"


def test_run_record_refuses_a_check_arm_with_screen_and_screen_with_check(home, tmp_path, capsys):
    scr = start(capsys, tmp_path, 3, 4)["screen"]
    code, out, err = lm(capsys, "run", "record", "--screen", scr, "--arm", "cand", *SOLO,
                        "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--no-commits", "--no-fit",
                        "--json")
    assert code == 1 and "--arm A" in out["error"]
    code, out, err = lm(capsys, "run", "record", "--screen", scr, "--check", "chk_01ABC", "--arm", "A", *SOLO,
                        "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--no-commits", "--no-fit",
                        "--json")
    assert code == 1 and "--screen" in out["error"] and "--check" in out["error"]


# ---------------------------------------------------------------- status and apply
def run_screen(capsys, home, cost, scr, usd_of, verdict_of=lambda on, rng: "pass", pairs=None, seed=0):
    """Record every pair (or the first `pairs`), each arm settling at `usd_of(on, rng)` dollars. The noise of a
    run comes from its own generator, keyed by the pair and the arm, so the arm order does not change it."""
    done = 0
    while pairs is None or done < pairs:
        code, nxt, err = lm(capsys, "screen", "next", scr, "--json")
        assert code == 0, err
        if nxt["next"] is None:
            break
        for arm in nxt["next"]["order"]:
            on = nxt["next"]["arms"][arm]["on"]
            rng = np.random.default_rng([seed, int(nxt["next"]["pair"][1:]), "AB".index(arm)])
            cost["usd"] = usd_of(on, rng)
            record(capsys, home, scr, arm, verdict_of(on, rng))
        done += 1


def test_status_ranks_the_edit_that_cuts_cost_first_and_names_the_best_subset(home, tmp_path, capsys, cost):
    scr = start(capsys, tmp_path, 3, 8, "--pair-usd", "0")["screen"]
    run_screen(capsys, home, cost, scr, lambda on, rng: 0.6 * math.exp(-0.4 * ("e02-line" in on)
                                                                         + 0.03 * rng.standard_normal()), seed=4)
    code, st, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    assert st["schema"] == "loopmath.screen/1" and st["screen"] == scr
    assert st["scored_pairs"] == 8 and st["objective"] == "cpar"
    edits = st["edits"]
    assert edits[0]["name"] == "e02-line"
    top = edits[0]
    for k in ("cost", "success", "p_helps", "rank"):
        assert k in top
    assert top["cost"]["ratio"]["mean"] < 0.75 and top["p_helps"] > 0.95
    assert top["cost"]["ratio"]["lo"] <= top["cost"]["ratio"]["mean"] <= top["cost"]["ratio"]["hi"]
    best = st["best"]
    assert "e02-line" in best["on"]
    assert 0.25 < best["improvement"]["mean"] < 0.4  # 1 - exp(-0.4) = 0.33 cut in cost per accepted result
    assert best["improvement"]["lo"] <= best["improvement"]["mean"] <= best["improvement"]["hi"]
    assert isinstance(best["tried"], bool)
    assert st["success_ceiling"] is True  # every run passed: success cannot rank the edits
    nv = st["next_pair"]
    assert "value_usd" in nv and nv["price_usd"] == 0
    code, text, err = lm(capsys, "screen", "status", scr)
    assert code == 0 and "e02-line" in text and "best" in text


def test_apply_writes_the_base_plus_the_best_edits(home, tmp_path, capsys, cost):
    scr = start(capsys, tmp_path, 3, 8)["screen"]
    run_screen(capsys, home, cost, scr,
               lambda on, rng: 0.6 * math.exp(-0.5 * ("e01-line" in on) + 0.4 * ("e03-line" in on)))
    out_dir = tmp_path / "best"
    code, out, err = lm(capsys, "screen", "apply", scr, "--best", "--out", str(out_dir), "--json")
    assert code == 0, err
    assert out["on"] == ["e01-line"]
    text = (out_dir / "CLAUDE.md").read_text()
    assert "- rule 1\n" in text and "- rule 3\n" not in text and "- rule 2\n" not in text
    assert out["harness"].startswith("sha256:")
    code, out, err = lm(capsys, "screen", "apply", scr, "--best", "--out", str(out_dir), "--json")
    assert code == 1 and "not empty" in out["error"]
    code, out, err = lm(capsys, "screen", "apply", scr, "--on", "e02-line,e03-line", "--out",
                        str(tmp_path / "chosen"), "--json")
    assert code == 0 and out["on"] == ["e02-line", "e03-line"]


def test_next_gives_a_named_pair_for_a_driver_with_several_in_flight(home, tmp_path, capsys):
    scr = start(capsys, tmp_path, 3, 4)["screen"]
    code, out, err = lm(capsys, "screen", "next", scr, "--pair", "p03", "--json")
    assert code == 0, err
    rec = json.loads((home / "screens" / f"{scr}.json").read_text())
    row = rec["pairs"][2]["row"]
    assert out["next"]["pair"] == "p03" and out["next"]["row"] == row
    assert out["next"]["arms"]["A"]["on"] == [f"e{j + 1:02d}-line" for j, x in enumerate(row) if x > 0]
    assert Path(out["next"]["arms"]["B"]["tree"]).is_dir()


def test_status_predicts_a_subset_nobody_ran(home, tmp_path, capsys, cost):
    """Edits 1 and 2 each cut cost by 20%. The subset of both (which may never have run as an arm) is predicted
    at about 1 - 0.8^2 = 36%, and an unknown edit is refused."""
    scr = start(capsys, tmp_path, 3, 8)["screen"]
    run_screen(capsys, home, cost, scr, lambda on, rng: 0.5 * 0.8 ** (("e01-line" in on) + ("e02-line" in on))
               * math.exp(0.02 * rng.standard_normal()), seed=3)
    code, st, err = lm(capsys, "screen", "status", scr, "--on", "e01-line,e02-line", "--json")
    assert code == 0, err
    sub = st["subset"]
    assert sub["on"] == ["e01-line", "e02-line"] and isinstance(sub["tried"], bool)
    assert 0.3 < sub["improvement"]["mean"] < 0.42
    assert sub["improvement"]["lo"] <= 0.36 <= sub["improvement"]["hi"]
    code, out, err = lm(capsys, "screen", "status", scr, "--on", "e09-line", "--json")
    assert code == 1 and "e09-line" in out["error"]


def test_status_before_any_pair_says_so_and_prints_the_next_pair(home, tmp_path, capsys):
    scr = start(capsys, tmp_path, 3, 4)["screen"]
    code, st, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    assert st["scored_pairs"] == 0 and st["best"] is None and st["next"]["pair"] == "p01"


def test_status_with_every_pair_tied_and_no_cost_says_nothing_ranks_the_edits(home, tmp_path, capsys, monkeypatch):
    scr = start(capsys, tmp_path, 3, 4)["screen"]
    monkeypatch.setattr(finish_mod, "default_settle", lambda: lambda doc: (doc, {"unmatched": {}}))  # no log matched
    for _ in range(4):
        record(capsys, home, scr, "A")
        record(capsys, home, scr, "B")
    code, st, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    tied = [w for w in st["warnings"] if w.startswith("every pair tied on success")]
    assert len(tied) == 1 and "nothing here ranks the edits" in tied[0] and "ranking is on cost" not in tied[0]
    assert st["best"]["on"] == []


# ---------------------------------------------------------------- end to end, known effects
E2E_SEED = int(os.environ.get("SCREEN_E2E_SEED", "12"))  # reruns the two end-to-end screens on other noise


def test_simulated_screen_recovers_known_cost_effects_at_the_success_ceiling(home, tmp_path, capsys, cost):
    """Every run passes, so success sits at its ceiling. 8 edits, 12 pairs, run-to-run noise of 0.2 in log cost.
    Edit 3 cuts cost by 35%, edit 6 adds 30%, the others do nothing. The ranked table puts edit 3 first, the best
    subset holds edit 3 and leaves out edit 6, and the predicted cut is near the true 35%."""
    scr = start(capsys, tmp_path, 8, 12, "--seed", "7")["screen"]
    log_effect = {"e03-line": math.log(0.65), "e06-line": math.log(1.3)}

    def usd(on, rng):
        return 0.5 * math.exp(sum(log_effect.get(e, 0.0) for e in on) + 0.2 * rng.standard_normal())

    run_screen(capsys, home, cost, scr, usd, seed=E2E_SEED)
    code, st, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    assert st["success_ceiling"] is True and st["stop"] == "cap"
    by = {e["name"]: e for e in st["edits"]}
    assert st["edits"][0]["name"] == "e03-line"
    assert by["e03-line"]["p_helps"] > 0.8 and by["e06-line"]["p_helps"] < 0.35
    assert by["e06-line"]["cost"]["ratio"]["mean"] > 1.05
    best = st["best"]["on"]
    assert "e03-line" in best and "e06-line" not in best
    assert 0.15 < st["best"]["improvement"]["mean"] < 0.55


def test_simulated_screen_keeps_an_edit_that_breaks_tasks_out_of_the_best(home, tmp_path, capsys, cost):
    """8 edits, 24 pairs. Edit 8 lowers the pass rate from 0.95 to 0.45, edit 3 cuts cost by 35%, the others do
    nothing. Edit 8 reads as hurting success, and the best subset holds edit 3 and leaves out edit 8."""
    scr = start(capsys, tmp_path, 8, 24, "--seed", "7")["screen"]

    def usd(on, rng):
        return 0.5 * math.exp(math.log(0.65) * ("e03-line" in on) + 0.2 * rng.standard_normal())

    def verdict(on, rng):
        return "pass" if rng.random() < 0.95 - 0.5 * ("e08-line" in on) else "fail"

    run_screen(capsys, home, cost, scr, usd, verdict, seed=E2E_SEED)
    code, st, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    by = {e["name"]: e for e in st["edits"]}
    assert by["e08-line"]["success"]["mean"] < -0.1 and by["e08-line"]["p_helps"] < 0.3
    assert by["e08-line"]["rank"] == 8
    best = st["best"]["on"]
    assert "e03-line" in best and "e08-line" not in best
