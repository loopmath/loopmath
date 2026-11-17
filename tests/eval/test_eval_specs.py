"""The catalog registry of `eval simulate` specs: `specs.names`, `specs.title`, `specs.get`."""

from __future__ import annotations

import pytest

from loopmath.eval import contracts as C
from loopmath.eval import simulate as S
from loopmath.eval import specs

# The catalog named by the acceptance checks, and the rest of the list in the simulate section of the design.
ACCEPTANCE = ("binary-checks", "levels-3", "levels-5", "case-items", "mixed-rubric", "gated", "pairs",
              "overall-vs-criteria", "own-family", "reference-family", "panel-vs-per-component", "steps-swap", "linear")
ALSO = ("one-component", "halo", "cheap-vs-strong-judge")


def test_the_catalog_holds_every_named_spec_the_basic_ones_first():
    names = specs.names()
    assert set(ACCEPTANCE) | set(ALSO) <= set(names) and len(set(names)) == len(names)
    assert set(names) - set(ACCEPTANCE) - set(ALSO) == {"case-items-uneven"}
    assert names[:4] == ["linear", "one-component", "binary-checks", "levels-3"]
    names.append("changed")
    assert "changed" not in specs.names()


@pytest.mark.parametrize("name", specs.names())
def test_a_title_is_one_plain_line(name):
    title = specs.title(name)
    assert title and title == title.strip() and "\n" not in title and len(title) <= 90
    assert chr(0x2014) not in title and not title.endswith(".")
    assert C._Check("names").ident(name, "name") == name


def test_get_returns_a_new_spec_with_its_name_and_title():
    one, two = specs.get("linear"), specs.get("linear")
    assert one == two and one is not two and one["arms"] is not two["arms"]
    assert one["schema"] == C.SCHEMA_SIM and one["name"] == "linear" and one["title"] == specs.title("linear")
    one["arms"][0]["mean"]["depth"] = 0.9
    one["judges"]["judges"][1]["model"]["family"] = "changed"
    assert specs.get("linear") == two


def test_inputs_scales_the_study_and_leaves_the_rest():
    small, usual = specs.get("pairs", inputs=7), specs.get("pairs")
    assert small["inputs"] == 7 and usual["inputs"] == 40
    assert {**small, "inputs": 40} == usual
    assert S.generate(small, 0).truth["counts"]["inputs"] == 7


def test_an_unknown_name_is_a_key_error_that_lists_the_catalog():
    with pytest.raises(KeyError) as caught:
        specs.get("no-such-spec")
    assert "no-such-spec" in str(caught.value) and "linear, one-component" in str(caught.value)
    with pytest.raises(KeyError):
        specs.title("no-such-spec")
