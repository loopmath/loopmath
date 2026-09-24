"""Tests for the standalone experiment explanation renderer."""

from loopmath.explain import ExperimentCandidate, render


def test_render_experiment_candidate_pitch():
    candidate = ExperimentCandidate(
        configuration_a="luna-low",
        configuration_b="sol-medium",
        upside_per_accepted=7.5,
        price=125,
        reuse_volume=240,
    )

    pitch = render(candidate)

    assert "luna-low" in pitch
    assert "sol-medium" in pitch
    assert "$125.00" in pitch
    assert "$1,800.00" in pitch


def test_rendered_pitch_uses_report_vocabulary():
    candidate = ExperimentCandidate("current", "proposed", 2, 10, 5)

    pitch = render(candidate).lower()

    forbidden = (
        "knowledge gradient",
        "posterior",
        "prior",
        "experimental design",
        "value of information",
        "bandit",
        "arms",
    )
    assert not any(term in pitch for term in forbidden)
