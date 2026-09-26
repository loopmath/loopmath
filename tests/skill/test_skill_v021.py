"""The 0.2.1 skill items (FINDINGS-0.2 F6, I17, I18, I20, P4, P5 and the recommend JSON contract of lane 21M):
the plan skill outside a git repo, the overlap with the shipped prior said once, `run import --brief`, one
payback number, the numbered options with `most_likely` and the chance within several attempts, the rescue
line, and a pasted `option N: LABEL`."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from loopmath.priors import overlap_note
from loopmath.skill import install as inst
from skill_texts import SH_BLOCK

PLAN = inst.skill_text("loopmath-plan-task")
IMPORT = inst.skill_text("loopmath-import-runs")
FIT = inst.skill_text("loopmath-update-fit")
REF = inst.reference_text()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _git_lines(text: str) -> list[str]:
    return [ln.strip() for block in SH_BLOCK.findall(text) for ln in block.splitlines() if ln.strip().startswith("git ")]


# ---------------------------------------------------------------- F6
def test_plan_takes_the_repo_from_the_users_words_and_works_without_git():
    flat = _flat(PLAN)
    assert 'When the user names the repo ("in ale-bench", "for acme/api"), `REPO` is that name, lowercased.' in flat
    assert "leave out `--base-commit SHA` in every command below and say so in one line" in flat
    assert '("No git checkout here, so the run has no base commit.")' in flat
    assert "Leave out `--base-commit SHA` with no git checkout." in flat  # run start too
    assert "git rev-parse HEAD; git remote get-url origin" not in PLAN  # 0.2.0's command, three fatal errors outside git


def test_every_git_command_in_the_plan_skill_drops_its_errors():
    lines = _git_lines(PLAN)
    assert lines == ["git rev-parse --show-toplevel HEAD 2>/dev/null", "git remote get-url origin 2>/dev/null"]


def test_the_git_commands_print_nothing_at_all_outside_a_repo(tmp_path):
    bash = shutil.which("bash") or pytest.skip("no bash on PATH")
    git = shutil.which("git") or pytest.skip("no git on PATH")
    env = {"PATH": f"{git.rsplit('/', 1)[0]}:/usr/bin:/bin", "HOME": str(tmp_path), "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    folder = tmp_path / "ale-bench-work"
    folder.mkdir()
    for line in _git_lines(PLAN):
        res = subprocess.run([bash, "--noprofile", "--norc", "-c", line], cwd=folder, env=env, capture_output=True,
                             text=True, stdin=subprocess.DEVNULL)
        assert res.stdout == "" and res.stderr == "", (line, res.stdout, res.stderr)


def test_the_reference_says_where_the_repo_and_base_commit_come_from():
    flat = _flat(REF)
    assert 'the repo or benchmark the user names ("in ale-bench" is `ale-bench`), lowercased' in flat
    assert "With no git checkout or no commit yet, leave the flag out everywhere and tell the user once" in flat


# ---------------------------------------------------------------- I17
def test_import_says_the_overlap_line_once_in_its_summary():
    flat = _flat(IMPORT)
    line = overlap_note({"rq1": 44})
    assert line == "44 of your runs are also in the shipped rq1 prior (same runs): fits use your copies"
    assert f'(for example "{line}")' in flat and "keep it for the summary, where it is said once, word for word" in flat
    summary = IMPORT.split("## 6. Summary", 1)[1].split("## Never", 1)[0]
    assert summary.count(line) == 1
    assert "The fit's `shipped_overlap` is the same overlap as `overlap_note`: do not name it a second time." in flat
    assert "If stderr says store runs share ids" not in IMPORT  # the fit prints nothing there (I17)


@pytest.mark.parametrize("counts, n, source", [({"rq1": 44}, "44", "rq1"), ({"e0": 1}, "1", "e0")])
def test_update_fit_builds_the_same_line_as_status(counts, n, source):
    flat = _flat(FIT)
    assert "`dropped`, `shipped_overlap`." in flat
    many = re.search(r'"(N of your runs are also in the shipped SOURCE prior \(same runs\): fits use your copies)"', flat)
    one = re.search(r'"(1 of your runs is also in the shipped SOURCE prior \(same run\): fits use your copy)"', flat)
    assert many and one
    template = one.group(1) if n == "1" else many.group(1)
    assert template.replace("N", n, 1).replace("SOURCE", source) == overlap_note(counts)
    assert '"the shipped prior (rq1 40, e0 4)"' in flat
    assert overlap_note({"rq1": 40, "e0": 4}).startswith("44 of your runs are also in the shipped prior (rq1 40, e0 4)")
    assert "the summary says one line, once" in flat and "do not name it again" in flat


# ---------------------------------------------------------------- I18
def test_import_uses_the_brief_form_and_reports_failures_with_file_and_error():
    assert "loopmath run import DIR --finish --no-fit --json --brief" in IMPORT
    flat = _flat(IMPORT)
    assert "Read `imported`, `failed`, `failures[]` (`file`, `error`), `more_failures`, `overlap_note`, `next`." in flat
    step3 = IMPORT.split("## 3. Import all", 1)[1].split("## 4. Fit once", 1)[0]
    assert "Report each failure with its file and error" in flat and "`files[]`" not in step3
    ref = _flat(REF)
    assert "`loopmath run import DIR --finish --no-fit --json --brief`" in ref
    assert "`failures[]` (`file`, `error`), `more_failures`, `shipped_overlap`, `overlap_note`, `fit`, `next`" in ref


# ---------------------------------------------------------------- I20
def test_the_plan_skill_shows_payback_runs_as_given():
    flat = _flat(PLAN)
    assert "`gain_per_future_run_usd`, `payback_runs`)" in flat
    assert "the payback: `payback_runs` similar tasks, as given (leave it out when it is null;" in flat
    ref = _flat(REF)
    assert ("`payback_runs` (similar runs until the extra run pays for itself: `price_now_usd` divided by "
            "`gain_per_future_run_usd`, rounded up, at least 1; null when the gain is 0; the number `message` says)") in ref


# ---------------------------------------------------------------- 21M contract and P5
def test_the_choices_are_numbered_by_option_with_both_chances_apart():
    flat = _flat(PLAN)
    assert "in the order of `choices[]`, numbered by `option` (without `option`, count from 1)" in flat
    assert ("then, when `p_accepted_within.attempts` is above 1, the chance within that many attempts "
            "(`p_accepted_within.mean`) as a second number (\"69% in one run (40 to 90%), 84% within 3 attempts\")") in flat
    assert "Keep the two chances apart; never merge them into one number." in flat
    assert 'Name the choice with `key` `most_likely` "Most likely to reach the target".' in flat
    example = [ln for ln in PLAN.splitlines() if re.match(r"> \d\. ", ln)]
    assert [ln[2] for ln in example] == ["1", "2", "3", "4", "5"] and "(1 to 5, or describe another workflow)" in PLAN
    assert any(ln.startswith("> 4. Most likely to reach the target:") for ln in example)
    assert all(re.search(r": \d+% in one run \(\d+ to \d+%\), \d+% within 3 attempts, \$", ln)
               for ln in example if "per accepted result" in ln)


def test_the_rescue_line_quotes_text_else_builds_it_else_falls_back():
    flat = _flat(PLAN)
    assert "When `rescue.text` is present, quote it word for word." in flat
    built = ('"Run cost is what the agents cost for one run; cost per accepted result adds the expected cost of fixing a '
             'miss by retrying with RESCUE, each extra attempt having half the chance of the one before, up to N attempts."')
    assert f"Else, when `rescue.kind` is `retry`: {built}, with `rescue.of` for RESCUE and `rescue.max_attempts` for N" in flat
    assert ("Else: cost per accepted result counts `expected_rescue_usd`, the expected cost of redoing a failed run "
            "with the reference workflow (`rescue`)") in flat  # the 0.2.0 wording


def test_a_pasted_option_maps_to_one_choice():
    flat = _flat(PLAN)
    assert "The planning page's \"Copy option\" button copies `option N: LABEL`" in flat
    assert "`option N: LABEL`: the choice whose `label` is LABEL, whatever its number." in flat
    assert ("When no choice has that label, the page came from another recommendation: say so in one line, show the "
            "current choices and ask the question once more. Never start a workflow other than the one pasted, even "
            "when option N exists;") in flat
    assert "when no label matches, the choice whose `option` is N" not in flat  # review 21A N1: no fallback by number
    assert "never start another workflow in its place" in _flat(REF) and "(else that `option`)" not in REF
    assert "`option N` or a number: the choice whose `option` is N (without `option`, the Nth in `choices[]`)" in flat
    for name, key in (("recommended", "goal"), ("pair", "pair"), ("most likely", "most_likely"),
                      ("cheapest", "cheapest_run")):
        assert f'"{name}" is `{key}`' in flat
    assert '"usual" or "reference" is `reference`' in flat
    assert "When no recommendation was made in this conversation, do steps 1 and 2 first, then map." in flat
    assert ("`workflow CFG: LABEL` (the page's copy for a workflow that is not one of the numbered options): that "
            "workflow, started with `--config CFG` (section 4).") in flat
    assert ('loopmath run start --rec REC --type TYPE --repo REPO --title "TITLE" --config CFG --source alternative '
            "--base-commit SHA --json") in " ".join(PLAN.replace("\\\n", " ").split())
    assert "For a row that is not a numbered option it copies `workflow CFG: LABEL`" in _flat(REF)
    desc = PLAN.split("---\n", 2)[1]
    assert "pastes an option from a loopmath planning page (option 2: ...)" in desc
    assert "`option N: ...` pasted from a planning page | `loopmath-plan-task`" in inst.skill_text("loopmath")


def test_the_reference_documents_the_new_recommend_fields():
    ref = _flat(REF)
    for needle in ("`key` (`goal`, `pair`, `reference`, `most_likely`, `cheapest_run`)",
                   "Up to 5 choices. Their order in `choices[]` is the order to number them in: each has `option`",
                   "`most_likely` (with a score target): the workflow with the highest chance to reach the target.",
                   "`p_accepted_within` on each choice (and in each `numbers` object): `mean`, `lo`, `hi`, `attempts`",
                   "the stored recommendation and the page data, not in `--brief`", '`{"50": [lo, hi], "80": [...], "90": [...], "95": [...]}`',
                   "`kind` `retry` (the default)", "`decay` (0.5)", "`max_attempts` (3)", "`min_chance` (0.7)",
                   "`p_accepted` is the chance the retries fix a miss", "`text` is one plain sentence about it: quote it word for word",
                   "Older kinds: `redo_usual`"):
        assert needle in ref, needle
