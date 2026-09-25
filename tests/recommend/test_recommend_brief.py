"""`recommend --json --brief` and `page` (lane 2D): the short object an agent reads before its one
question, and the planning page's path in the same JSON."""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.recommend.commands import BRIEF_KEYS

from test_recommend_command import env, make_belief  # noqa: F401  (env is a fixture)

ARGS = ["recommend", "--type", "feature", "--repo", "acme/app", "--feature", "size=80", "--json"]


def _run(capsys, *extra) -> tuple[dict, str, str]:
    assert cli.main([*ARGS, *extra]) == 0
    out, err = capsys.readouterr()
    return json.loads(out), out, err


def _stored(env, rec: str) -> dict:
    home, _ = env
    return json.loads((home / "recs" / f"{rec}.json").read_text())


def test_brief_keeps_the_listed_keys_as_the_full_json_has_them(env, capsys):
    """Compared with the stored recommendation of the same call, which is the full JSON plus candidates."""
    brief, brief_text, _ = _run(capsys, "--brief")
    full = {k: v for k, v in _stored(env, brief["rec"]).items() if k != "candidates"}
    assert list(brief)[0] == "schema" and brief["schema"] == full["schema"] == "loopmath.recommend/2"
    assert set(brief) == {"schema", *(k for k in BRIEF_KEYS if k in full)}
    for key in set(brief) - {"reference"}:
        assert brief[key] == full[key], key
    assert brief["reference"] == {k: v for k, v in full["reference"].items() if k != "prediction"}
    assert brief["choices"] and all("key" in c for c in brief["choices"])
    assert "curve" not in brief and "alternatives" not in brief and "exploration" not in brief
    assert len(brief_text) < len(json.dumps(full)) / 2 and len(brief_text) < 10_000


def test_the_goal_strategy_survives_the_projection(env, capsys):
    """2A's goal.strategy (D119 Z2), on `goal` and on the goal choice, is carried through as it is."""
    brief, _, _ = _run(capsys, "--brief")
    full = _stored(env, brief["rec"])
    assert brief["goal"].get("strategy") == full["goal"].get("strategy")
    goal_choice = next(c for c in brief["choices"] if c["key"] == "goal")
    assert goal_choice.get("strategy") == next(c for c in full["choices"] if c["key"] == "goal").get("strategy")


def test_the_stored_recommendation_keeps_everything(env, capsys):
    brief, _, _ = _run(capsys, "--brief")
    stored = _stored(env, brief["rec"])
    assert "curve" in stored and "candidates" in stored and stored["choices"] == brief["choices"]


@pytest.mark.parametrize("extra", [[], ["--brief"]])
def test_page_is_the_path_html_wrote(env, capsys, tmp_path, extra):
    target = tmp_path / "plan.html"
    obj, out, err = _run(capsys, *extra, "--html", str(target))
    assert obj["page"] == str(target) and target.is_file()
    assert out.lstrip().startswith("{") and out.count(str(target)) == 1  # stdout stays one object
    assert err.strip().endswith(str(target))  # and the path still goes to stderr, as before
    obj, _, _ = _run(capsys, *extra)
    assert "page" not in obj


def test_brief_without_json_changes_nothing(env, capsys):
    assert cli.main([*ARGS[:-1]]) == 0
    text = capsys.readouterr().out
    assert cli.main([*ARGS[:-1], "--brief"]) == 0
    same = lambda t: [ln for ln in t.splitlines() if "rec_" not in ln and "tsk_" not in ln]  # noqa: E731
    assert same(capsys.readouterr().out) == same(text)
