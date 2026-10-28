"""`loopmath harness plan`: session counts, the cost point and range, time, residual df, warnings and preflight.
No model call: preflight runs stub `claude` and `herdr` binaries put first on PATH."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune import plan as planmod


class Prices:
    """A price table with per-Mtok rates; models it lacks are unpriced."""

    def __init__(self, rates):
        self.rates = rates

    def rate(self, model):
        return self.rates.get(model)


PRICES = Prices({
    "claude-haiku-5-5": {"input": 0.10, "cache_read": 0.01, "cache_write": 0.20, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "cache_read": 0.10, "cache_write": 4.00, "output": 10.00},
    "claude-opus-5-5": {"input": 5.00, "cache_read": 0.50, "cache_write": 10.00, "output": 25.00},
})


def plan_of(*sets, editors=None, edits=None, prices=PRICES):
    return planmod.make_plan(configmod.resolve(sets=list(sets), editors=editors), edits=edits, prices=prices)


def test_default_plan_counts_and_cost():
    p = plan_of()
    assert p.sessions == {"executor": 15, "editor": 2, "judge": 1, "reflector": 1}
    assert (round(p.cost["point"], 2), round(p.cost["low"], 2), round(p.cost["high"], 2)) == (2.72, 1.72, 6.38)
    assert p.probes == 12 and p.edits == 10 and p.df == 3 and p.design == "pb12"
    assert p.stages == ["judge", "base", "signal", "edit", "design", "probe", "fit", "build", "final", "deliver",
                        "reflect"]
    assert p.sessions_high == {"executor": 15, "editor": 4, "judge": 3, "reflector": 2}
    assert p.caps == {"usd": 10.0, "runs": 40, "wall_s": 21600.0, "set_usd": 0.0}


def test_default_plan_prints_the_documented_lines():
    text = planmod.render(plan_of())
    assert "sessions: 15 executor runs, 2 editor sessions, 1 reflector, 1 test writer\n" in text
    assert "cost: 2.72 USD (1.72 to 6.38)\n" in text
    assert "design: pb12, 12 probe runs over up to 10 edits, residual df 3\n" in text
    assert "time: typical unknown; worst case" in text
    assert "caps: 10.00 USD, 40 runs, 6.0 h\n" in text


def test_judge_none_with_a_score_command():
    p = plan_of('judge.kind="none"', 'score.command="pytest -q"')
    assert p.sessions["judge"] == 0 and "judge" not in p.stages
    assert (round(p.cost["point"], 2), round(p.cost["low"], 2), round(p.cost["high"], 2)) == (2.38, 1.55, 5.36)
    assert "test writer" not in planmod.render(p)


def test_cost_by_role_is_count_times_prior():
    c = plan_of().cost["by_role"]
    assert c["executor"]["point"] == pytest.approx(15 * 0.040)
    assert c["executor"]["low"] == pytest.approx(15 * 0.040 * 0.5)
    assert c["executor"]["high"] == pytest.approx(15 * 0.040 * 3)
    assert c["editor"]["point"] == pytest.approx(2 * 0.765) and c["editor"]["high"] == pytest.approx(4 * 0.765)
    assert c["judge"]["point"] == pytest.approx(0.34) and c["reflector"]["point"] == pytest.approx(0.25)
    assert c["reflector"]["high"] == pytest.approx(0.50)


def test_another_model_scales_the_prior_by_the_blended_rate():
    p = plan_of('executor.model="claude-sonnet-5-5"')
    mix = planmod.TOKEN_MIX
    blend = {m: sum(mix[k] * r[k] for k in mix) for m, r in PRICES.rates.items()}
    ratio = blend["claude-sonnet-5-5"] / blend["claude-haiku-5-5"]
    assert p.cost["by_role"]["executor"]["point"] == pytest.approx(15 * 0.040 * ratio, rel=1e-3)
    assert not p.warnings or all("price row" not in w for w in p.warnings)


def test_a_model_with_no_price_row_is_priced_at_the_prior_and_warned():
    p = plan_of('executor.model="claude-unpriced-9"')
    assert p.cost["by_role"]["executor"]["point"] == pytest.approx(0.60)
    assert any("no price row for claude-unpriced-9" in w for w in p.warnings)


def test_the_default_worst_case_passes_the_wall_cap_and_says_so():
    p = plan_of()
    assert p.time["typical_s"] is None and p.time["worst_s"] > 6 * 3600
    assert p.time["waves"]["probe"] == 3 and p.time["waves"]["base"] == 1
    assert any("budget.wall" in w for w in p.warnings)
    assert not any("budget.wall" in w for w in plan_of('budget.wall="12h"').warnings)


@pytest.mark.parametrize("sets,executor,df", [
    (['deliver.rule="fit"'], 14, 3),
    (["final.runs=2"], 16, 3),
    (["base.runs=3"], 16, 4),
    (["screen.repeats=2"], 27, 15),
    (['screen.design="one-at-a-time"'], 13, 1),
    (['screen.design="fold-over"', "screen.pairs=8"], 19, 7),
    (["screen.base_rows=false"], 15, 1),
])
def test_counts_and_df_follow_the_design(sets, executor, df):
    p = plan_of(*sets)
    assert p.sessions["executor"] == executor and p.df == df


def test_full_design_and_edit_folders_set_the_edit_count():
    p = plan_of('screen.design="full"', "screen.max_edits=4")
    assert p.edits == 4 and p.probes == 16 and p.df == 18 - 5
    p = plan_of("edit.enabled=false", 'edit.from="mine"', edits=3)
    assert p.edits == 3 and p.sessions["editor"] == 0 and p.probes == 12


def test_low_df_and_no_edit_warn():
    assert any("residual degrees of freedom" in w for w in plan_of("base.runs=1", "screen.base_rows=false").warnings)
    p = plan_of("edit.enabled=false", 'edit.from="mine"', edits=0)
    assert p.probes == 0 and p.sessions["executor"] == 2 + 0
    assert any("measures the base only" in w for w in p.warnings)


def test_gate_adds_the_replicate_pairs_to_the_high_end():
    base, gate = plan_of(), plan_of('deliver.rule="gate"')
    assert gate.sessions["executor"] == base.sessions["executor"]
    assert gate.sessions_high["executor"] == base.sessions_high["executor"] + 6
    assert gate.cost["high"] == pytest.approx(base.cost["high"] + 6 * 0.040 * 3)
    assert "check" in gate.stages


def test_editor_count_and_rounds():
    p = plan_of(editors=3)
    assert p.sessions["editor"] == 3 and p.edits == 10
    p = plan_of(editors=1)
    assert p.edits == 5 and p.sessions["editor"] == 1
    p = plan_of("loop.rounds=2")
    assert p.sessions["executor"] == 2 + 2 * 13 + 1 and p.sessions["editor"] == 4 and p.sessions["reflector"] == 2


def test_reflector_after_fit_moves_the_stage():
    p = plan_of('reflector.when="after-fit"')
    assert p.stages.index("reflect") == p.stages.index("fit") + 1
    assert "reflect" not in plan_of("reflector.enabled=false").stages


def test_budget_warnings():
    assert any("high estimate" in w for w in plan_of("budget.usd=5").warnings)
    assert any("budget.runs" in w for w in plan_of("budget.runs=10").warnings)


def test_session_estimates_for_the_budget_guard():
    assert planmod.unit_cost("executor", "claude-haiku-5-5", PRICES) == pytest.approx(0.040)
    assert planmod.session_high("executor", "claude-haiku-5-5", PRICES) == pytest.approx(0.120)
    assert planmod.session_high("editor", "claude-opus-5-5", PRICES) == pytest.approx(1.53)
    assert planmod.unit_cost("executor", "", PRICES, {"executor": 0.6}) == 0.6
    assert planmod.session_high("executor", "", PRICES, {"executor": 0.6}) == pytest.approx(1.8)


def test_plan_record_is_json_ready():
    import json

    d = plan_of().to_dict()
    assert json.loads(json.dumps(d))["sessions"]["executor"] == 15
    assert d["cost"]["priors"]["executor"] == {"model": "claude-haiku-5-5", "usd": 0.04}


# ---------------------------------------------------------------------------------------------------- preflight


def stub(folder: Path, name: str, out: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(f"#!/bin/sh\necho '{out}'\n")
    p.chmod(0o755)
    return p


def checks_of(result):
    return {c["check"]: c for c in result}


@pytest.fixture
def bin_dir(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}/usr/bin{os.pathsep}/bin")
    return d


def test_preflight_passes_with_new_enough_binaries_and_a_600_token(tmp_path, bin_dir):
    stub(bin_dir, "claude", "2.1.301 (Claude Code)")
    stub(bin_dir, "herdr", "herdr 0.9.3")
    token = tmp_path / "token"
    token.write_text("not read\n")
    token.chmod(0o600)
    cfg = configmod.resolve(sets=[f'auth.token_file="{token}"', 'run.runner="herdr"'])
    got = checks_of(planmod.preflight(cfg, prices=PRICES))
    assert got["claude"]["ok"] and "2.1.301" in got["claude"]["line"]
    assert got["herdr"]["ok"] and got["token"]["ok"] and got["prices"]["ok"]


def test_preflight_fails_an_old_claude_and_a_missing_herdr(tmp_path, bin_dir):
    stub(bin_dir, "claude", "2.1.299 (Claude Code)")
    cfg = configmod.resolve(sets=['run.runner="herdr"'])
    got = checks_of(planmod.preflight(cfg, prices=PRICES))
    assert not got["claude"]["ok"] and "floor 2.1.301" in got["claude"]["line"]
    assert not got["herdr"]["ok"] and "not on PATH" in got["herdr"]["line"]
    assert not got["token"]["ok"] and "setup-token" in got["token"]["line"]


def test_preflight_token_file_mode_is_checked_without_reading_it(tmp_path, bin_dir):
    stub(bin_dir, "claude", "2.1.301")
    token = tmp_path / "token"
    token.write_text("secret-canary\n")
    token.chmod(0o644)
    cfg = configmod.resolve(sets=[f'auth.token_file="{token}"'])
    got = checks_of(planmod.preflight(cfg, prices=PRICES))
    assert not got["token"]["ok"] and "needs 600" in got["token"]["line"]
    assert "secret-canary" not in repr(got)
    token.chmod(0o000)
    try:
        got = checks_of(planmod.preflight(cfg, prices=PRICES))  # a file it cannot read is still checked by mode
        assert "mode 0" in got["token"]["line"]
    finally:
        token.chmod(0o600)


def test_preflight_api_key_env(bin_dir, monkeypatch):
    stub(bin_dir, "claude", "2.1.301")
    cfg = configmod.resolve(sets=['auth.mode="api-key-env"', 'auth.api_key_env="TUNE_TEST_KEY"'])
    monkeypatch.delenv("TUNE_TEST_KEY", raising=False)
    assert not checks_of(planmod.preflight(cfg, prices=PRICES))["token"]["ok"]
    monkeypatch.setenv("TUNE_TEST_KEY", "x")
    got = checks_of(planmod.preflight(cfg, prices=PRICES))["token"]
    assert got["ok"] and "x" not in got["line"].replace("TUNE_TEST_KEY", "").replace("is set", "")


def test_preflight_missing_price_rows(bin_dir):
    stub(bin_dir, "claude", "2.1.301")
    got = checks_of(planmod.preflight(configmod.resolve(), prices=Prices({})))["prices"]
    assert not got["ok"] and "claude-haiku-5-5" in got["line"] and "claude-sonnet-5-5" in got["line"]


def test_preflight_repo_and_base_ref(tmp_path, bin_dir):
    stub(bin_dir, "claude", "2.1.301")
    git = shutil.which("git", path="/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    if git is None:
        pytest.skip("no git")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run([git, "-C", str(repo), "init", "-q"], check=True, env=env)
    subprocess.run([git, "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "s"], check=True, env=env)

    def run(args, **kw):
        if args[0] == "git":
            args = [git, *args[1:]]
        return subprocess.run(args, **kw, env=env)

    ok = checks_of(planmod.preflight(configmod.resolve(), repo=repo, prices=PRICES, run=run))["repo"]
    assert ok["ok"]
    bad = configmod.resolve(sets=['task.base_ref="nonesuch"'])
    assert not checks_of(planmod.preflight(bad, repo=repo, prices=PRICES, run=run))["repo"]["ok"]


def test_preflight_finds_instruction_files_above_the_work_root(tmp_path, bin_dir):
    stub(bin_dir, "claude", "2.1.301")
    root = tmp_path / "outer" / "tunes"
    root.mkdir(parents=True)
    assert checks_of(planmod.preflight(configmod.resolve(), work_root=root, prices=PRICES))["work_root"]["ok"]
    (tmp_path / "outer" / "CLAUDE.md").write_text("# would load into every run\n")
    got = checks_of(planmod.preflight(configmod.resolve(), work_root=root, prices=PRICES))["work_root"]
    assert not got["ok"] and "CLAUDE.md" in got["line"]


def test_preflight_lines_print_in_the_plan(bin_dir):
    stub(bin_dir, "claude", "2.1.301")
    cfg = configmod.resolve()
    p = planmod.make_plan(cfg, prices=PRICES, checks=planmod.preflight(cfg, prices=PRICES))
    text = planmod.render(p)
    assert "preflight ok: claude 2.1.301" in text and "preflight FAIL: auth.token_file is not set" in text
    assert stat.S_IMODE(os.stat(bin_dir / "claude").st_mode) == 0o755
