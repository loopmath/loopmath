"""Composed shapes onboard names runs with (`team`, `team_review`, ...): `workflows show`, `diff` and `list`
explain them; the catalog, and so recommend, stay as they are (new-user test P3-19)."""

from __future__ import annotations

import json

from loopmath.cli import main
from loopmath.workflows.format import catalog
from loopmath.workflows.shapes import CATALOG_IDS, composed_shapes


def test_composed_shapes_are_outside_the_catalog():
    names = list(composed_shapes())
    assert {"team", "team_review"} <= set(names)
    assert not set(names) & set(CATALOG_IDS) and not set(names) & set(catalog())


def test_show_team(tmp_path, capsys):
    assert main(["workflows", "show", "team", "--home", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("team: ") and "composed, not in the catalog, so recommend does not offer it" in out
    assert "the nearest catalog shape is swarm" in out
    assert main(["workflows", "show", "team_review", "--home", str(tmp_path), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["origin"] == "composed" and doc["workflow"]["id"] == "team_review"


def test_list_and_diff(tmp_path, capsys):
    assert main(["workflows", "list", "--home", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "also composed, not in the catalog" in out and "team, team_review" in out
    assert main(["workflows", "list", "--home", str(tmp_path), "--json"]) == 0
    assert "team" in json.loads(capsys.readouterr().out)["composed"]
    assert main(["workflows", "diff", "team", "swarm", "--home", str(tmp_path)]) == 0
    capsys.readouterr()


def test_an_unknown_name_says_what_names_work(tmp_path, capsys):
    assert main(["workflows", "show", "nosuch", "--home", str(tmp_path)]) != 0
    assert "a composed shape (such as team or team_review)" in capsys.readouterr().err
