from loopmath.e0.arms import (
    build_attempt_table,
    build_session_table,
    canonicalize_attempt_model,
    project_family,
    session_arm,
)


def test_canonicalize_attempt_model():
    # the observed vocabulary from design doc 1b, all mapping to one arm
    assert canonicalize_attempt_model("claude-fable-5·xhigh") == "fable5·xhigh"
    assert canonicalize_attempt_model("fable5·xhigh") == "fable5·xhigh"
    assert canonicalize_attempt_model("fable·xhigh") == "fable5·xhigh"
    assert canonicalize_attempt_model("opus5.xhigh") == "fable5·xhigh".replace("fable5", "opus5")
    # sol5.6 keeps its dot; only the effort suffix splits
    assert canonicalize_attempt_model("sol5.6·max") == "sol5.6·max"
    assert canonicalize_attempt_model("opus5·1m") == "opus5·1m"
    # null-ish labels
    assert canonicalize_attempt_model(None) is None
    assert canonicalize_attempt_model("-") is None
    assert canonicalize_attempt_model("") is None


def test_session_arm():
    s = {
        "primary_model": "claude-opus-5",
        "reasoning_effort_signals": ["assistant.effort=medium(10)"],
    }
    assert session_arm(s) == "opus5·medium"
    assert session_arm({"primary_model": None}) is None


def test_project_family():
    assert project_family("/Users/x/Workspace/pitauri-t3") == project_family(
        "/Users/x/Workspace/pitauri-t7"
    )
    assert project_family("/Users/x/Workspace/video-analyzer") == project_family(
        "/Users/x/Workspace/video-orchestrator"
    )
    assert project_family(None) == "unknown"


def test_build_session_table(fake_sessions):
    from loopmath.e0.io import segment_sessions

    df = build_session_table(segment_sessions(fake_sessions)["main"])
    assert len(df) == 8
    assert set(df["arm"].dropna()) == {"opus5·medium", "fable5·xhigh"}
    assert df["accepted"].sum() == 6
    assert "cell" in df.columns


def test_build_attempt_table(fake_attempts):
    df = build_attempt_table(fake_attempts)
    assert len(df) == 10
    assert df["arm"].isna().sum() == 2  # null and dash labels
    assert set(df["arm"].dropna()) == {"opus5·medium", "fable5·xhigh", "opus5·xhigh"}
    # accepted needs done + verified/reported evidence
    assert not df[df["outcome_result"] == "rejected"]["accepted"].any()
