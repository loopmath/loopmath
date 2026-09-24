"""Labelling hooks in taskmodel: the answer schema, the instructions, validation, keyword guess."""

from __future__ import annotations

from loopmath import taskmodel
from loopmath.taskmodel import FEATURES, TASK_TYPES, guess_type, label_instructions, label_schema, validate_label


def _closed(schema: dict) -> None:
    """Every object is closed and requires every property (strict structured output)."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        for sub in schema["properties"].values():
            _closed(sub)
    if schema.get("type") == "array":
        _closed(schema["items"])


def test_schema_is_strict_and_lists_every_type_and_feature():
    schema = label_schema()
    _closed(schema)
    item = schema["properties"]["labels"]["items"]
    assert item["properties"]["type"]["enum"] == [*TASK_TYPES, "unknown"]
    assert set(item["properties"]["features"]["properties"]) == {spec.key for spec in FEATURES.values()}
    assert item["properties"]["subtype"] == {"type": "null"}


def test_schema_with_subtypes():
    item = label_schema(["hotfix", "flaky"])["properties"]["labels"]["items"]
    assert item["properties"]["subtype"]["enum"] == ["hotfix", "flaky", None]


def test_instructions_name_types_features_and_the_no_tool_rule():
    text = label_instructions(["hotfix"])
    for kind in TASK_TYPES:
        assert f"- {kind}:" in text
    for spec in FEATURES.values():
        assert f"- {spec.key} (" in text
    assert "hotfix" in text and "Do not use any tool" in text
    assert "Subtype: always null." in label_instructions()


def test_validate_label_keeps_known_values_and_drops_the_rest():
    raw = {"id": "grp_x", "type": "Bug_Fix", "subtype": "flaky", "confidence": 1.7,
           "title": "  Fix   the\nparser  " + "x" * 200,
           "features": {"size": "m", "lang": "Python", "has_tests": "unknown", "color": "blue"}}
    label, problems = validate_label(raw, ["hotfix"])
    assert label["type"] == "bug_fix"
    assert label["subtype"] is None and problems == ["subtype 'flaky' not in config subtypes"]
    assert label["confidence"] == 1.0
    assert label["title"].startswith("Fix the parser x") and len(label["title"]) == taskmodel.TITLE_MAX
    assert "color" not in label["features"] and "extra:color" not in label["features"]
    assert "has_tests" not in label["features"]
    assert label["features"]["size"] == "m"


def test_validate_label_subtype_in_list():
    label, problems = validate_label({"type": "feature", "subtype": "hotfix", "confidence": 0.5,
                                      "features": {}, "title": "t"}, ["hotfix"])
    assert label["subtype"] == "hotfix" and problems == []


def test_validate_label_unknown_and_invalid():
    assert validate_label({"type": "unknown"}) == (None, ["type unknown"])
    assert validate_label({"type": "poetry"})[0] is None
    assert validate_label("bug_fix") == (None, ["not an object"])
    label, problems = validate_label({"type": "docs", "confidence": "high"})
    assert label["confidence"] == 0.0 and "confidence missing" in problems


def test_guess_type_order_and_misses():
    assert guess_type("Fix the failing unit tests") == ("bug_fix", taskmodel.GUESS_CONFIDENCE)
    assert guess_type("Add unit tests for the parser")[0] == "tests"
    assert guess_type("Update the README")[0] == "docs"
    assert guess_type("Refactor the store module")[0] == "refactor"
    assert guess_type("Why does the fit take so long?")[0] == "research"
    assert guess_type("Implement a dark mode toggle")[0] == "feature"
    assert guess_type("hello there") == (None, 0.0)
    assert guess_type(None) == (None, 0.0)
