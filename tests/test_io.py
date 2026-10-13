from loopmath.e0.io import (
    dominant_effort,
    load_corpus,
    parse_effort_signals,
    segment_sessions,
)


def test_load_corpus(corpus_dir):
    corpus = load_corpus(corpus_dir)
    assert len(corpus["sessions"]) == 12
    assert len(corpus["dag_attempts"]) == 10
    assert len(corpus["dag_runs"]) == 1
    assert corpus["join"] == []


def test_parse_effort_signals():
    sigs = ["assistant.effort=xhigh(695)", "thinking_blocks(208)", "assistant.effort=medium(3)"]
    assert parse_effort_signals(sigs) == {"xhigh": 695, "medium": 3}
    assert parse_effort_signals(None) == {}
    assert parse_effort_signals(["thinking_blocks(5)"]) == {}


def test_dominant_effort():
    s = {"reasoning_effort_signals": ["assistant.effort=medium(3)", "assistant.effort=xhigh(9)"]}
    assert dominant_effort(s) == "xhigh"
    assert dominant_effort({"reasoning_effort_signals": []}) is None


def test_segmentation_rule(fake_sessions):
    seg = segment_sessions(fake_sessions)
    assert len(seg["main"]) == 8
    assert len(seg["fleet_stubs"]) == 3
    assert len(seg["subagents"]) == 1
    # the rule is exact: primary_model null AND error_events >= 1
    for s in seg["fleet_stubs"]:
        assert s["primary_model"] is None
        assert s["outcome"]["error_events"] >= 1
    # a null-model session WITHOUT errors is not a stub
    no_model = dict(fake_sessions[0], primary_model=None)
    no_model["outcome"] = dict(no_model["outcome"], error_events=0)
    assert segment_sessions([no_model])["main"] == [no_model]
