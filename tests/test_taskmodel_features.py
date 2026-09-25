"""Declared task features and the run horizon (lane 2C, I14; spec 04 section 1)."""

from __future__ import annotations

import pytest

from loopmath import taskmodel as T

DECLARED = {
    "min_tasks": 3,
    "domain": {"values": ["web", "cli"], "description": "What the task is for."},
    "grid": {"kind": "bool", "fill": "labeller", "types": ["feature", "bug_fix"]},
    "n": {"kind": "number", "edges": [100, 1000], "labels": ["small", "mid", "big"]},
}


def test_builtins_unchanged():
    fs = T.FeatureSet()
    assert list(fs.specs) == list(T.FEATURES) and fs.min_tasks == T.MIN_TASKS_DEFAULT == 2
    raw = {"size": "120", "touches": "7", "lang": "Python", "has_tests": "maybe", "Other": "X"}
    assert T.normalize_features(raw) == {"size": "m", "touches": "many", "lang": "python",
                                         "has_tests": "unknown", "extra:other": "x"}
    assert T.normalize_features(T.normalize_features(raw)) == T.normalize_features(raw)  # idempotent


def test_declared_kinds_normalize():
    fs = T.FeatureSet.from_config(DECLARED)
    assert fs.min_tasks == 3 and [s.key for s in fs.custom] == ["domain", "grid", "n"]
    got = fs.normalize({"domain": "WEB", "grid": "Yes", "n": "2500", "size": "s", "horizon_s": "8h", "note": "hi"})
    assert got == {"domain": "web", "grid": "yes", "n": "big", "size": "s", "horizon_s": "28800", "extra:note": "hi"}
    assert fs.normalize({"domain": "mobile", "grid": "perhaps", "n": "lots", "horizon_s": "none"}) == {
        "domain": "unknown", "grid": "unknown", "n": "unknown", "horizon_s": "none"}  # a given open-ended horizon
    assert fs.normalize({"horizon_s": "soon"}) == fs.normalize({"horizon_s": ""}) == {}
    stored = fs.store({"domain": "Mobile", "grid": True, "n": "2500", "size": "120", "horizon_s": "2h", "note": "hi"})
    assert stored == {"domain": "mobile", "grid": "yes", "n": "2500", "size": "m", "horizon_s": "7200",
                      "extra:note": "hi"}  # declared keys raw, built-ins bucketed as before
    assert fs.normalize(stored) == fs.normalize(fs.store(stored)) == {
        "domain": "unknown", "grid": "yes", "n": "big", "size": "m", "horizon_s": "7200", "extra:note": "hi"}
    assert fs.normalize({"n": "99.5", "grid": True})["n"] == "small"
    assert fs.normalize({"n": "100"})["n"] == "mid"  # an edge starts the next bucket
    assert fs.normalize({"n": "mid"})["n"] == "mid"  # a label is accepted as given
    assert fs.model_features({"domain": "web", "grid": "no", "n": "x", "horizon_s": "7200", "z": "1"}) == (
        ("domain", "web"), ("grid", "no"))
    assert [s.key for s in fs.labeller_specs()][-1] == "grid"  # custom keys default to orchestrator


def test_json_roundtrip_and_digest():
    fs = T.FeatureSet.from_config(DECLARED)
    back = T.FeatureSet.from_json(fs.to_json())
    assert back.to_json() == fs.to_json() and back.digest() == fs.digest()
    assert T.FeatureSet.from_json(None).digest() == T.FeatureSet().digest() != fs.digest()
    assert T.FeatureSet.from_json({"custom": [{"key": "size"}]}).custom == ()  # a bad record reads as the built-ins


@pytest.mark.parametrize("table, words", [
    ({"size": {"values": ["a"]}}, "built-in"),
    ({"horizon_s": {}}, "horizon"),
    ({"extra_x": {}}, "extra"),
    ({"Bad-Key": {}}, "lowercase"),
    ({"d": {"values": ["a", "a"]}}, "twice"),
    ({"d": {"values": ["unknown"]}}, "always allowed"),
    ({"d": {"values": [f"v{i}" for i in range(13)]}}, "at most 12"),
    ({"d": {"kind": "enum"}}, "kind"),
    ({"d": {"fill": "human"}}, "fill"),
    ({"d": {"types": ["chore"]}}, "task type"),
    ({"d": {"color": "red"}}, "unknown field"),
    ({"d": {"kind": "number", "edges": [5, 1], "labels": ["a", "b", "c"]}}, "ascending"),
    ({"d": {"kind": "number", "edges": [1], "labels": ["a"]}}, "needs 2 labels"),
    ({"d": {"kind": "number"}}, "at least one"),
    ({"d": {"edges": [1], "labels": ["a", "b"]}}, "number only"),
    ({"d": {"kind": "bool", "values": ["a"]}}, "category only"),
    ({"min_tasks": 0}, "min_tasks"),
    ({f"k{i}": {} for i in range(17)}, "at most 16"),
])
def test_bad_declarations(table, words):
    with pytest.raises(T.FeatureConfigError, match=words):
        T.FeatureSet.from_config(table)


@pytest.mark.parametrize("text, secs", [("8h", 28800), ("90m", 5400), ("2.5h", 9000), ("45s", 45), ("7200", 7200),
                                        (7200, 7200), ("1 day", 86400), ("none", None), ("", None), ("open", None),
                                        (None, None)])
def test_parse_horizon(text, secs):
    assert T.parse_horizon(text) == secs


@pytest.mark.parametrize("text", ["soon", "8x", "-2h", True, "h"])
def test_parse_horizon_rejects(text):
    with pytest.raises(ValueError):
        T.parse_horizon(text)


def test_horizon_text():
    assert [T.horizon_text(s) for s in (None, 7200, 5400, 1800, 45)] == ["open-ended", "2 h", "1.5 h", "30 min", "45 s"]


def test_labeller_sees_declared_keys_in_scope():
    fs = T.FeatureSet.from_config(DECLARED)
    schema = T.label_schema(["api"], fs)
    props = schema["properties"]["labels"]["items"]["properties"]["features"]["properties"]
    assert props["grid"] == {"type": "string", "enum": ["yes", "no", "unknown"]}
    assert "domain" not in props and "n" not in props  # orchestrator keys are not the labeller's
    assert set(T.label_schema(["api"])["properties"]["labels"]["items"]["properties"]["features"]["properties"]) == set(
        T.FEATURES)
    text = T.label_instructions(["api"], fs)
    assert "- grid (yes|no|unknown)" in text and "Only for feature, bug_fix tasks; else unknown." in text
    item = {"id": "g1", "type": "feature", "subtype": "api", "confidence": 0.8, "title": "t",
            "features": {"grid": "yes", "size": "s", "domain": "web"}}
    label, problems = T.validate_label(item, ["api"], fs)
    assert label["features"] == {"grid": "yes", "size": "s"} and problems == []
    label, _ = T.validate_label({**item, "type": "docs"}, ["api"], fs)
    assert label["features"] == {"size": "s"}  # grid is declared for feature and bug_fix only
    label, _ = T.validate_label(item, ["api"])
    assert label["features"] == {"size": "s"}  # the built-in set drops undeclared keys, as before


def test_task_types_payload_additive():
    fs = T.FeatureSet.from_config(DECLARED)
    payload = T.task_types_payload(["api"], fs)
    keys = [f["key"] for f in payload["features"]]
    assert keys == [*T.FEATURES, "domain", "grid", "n"]
    first = payload["features"][0]
    assert {"key", "values", "description"} <= set(first) and first["builtin"] is True
    n = payload["features"][-1]
    assert n["kind"] == "number" and n["values"] == ["small", "mid", "big"] and n["edges"] == [100.0, 1000.0]
    assert payload["horizon"]["key"] == "horizon_s" and payload["horizon"]["flag"] == "--horizon"
