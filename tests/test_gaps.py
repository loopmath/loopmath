"""Tests for typed coverage-gap detection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from loopmath.gaps import ConfoundedPair, ObserverLimited, UnvisitedArm, detect_gaps


FORBIDDEN = [
    "knowledge gradient",
    "posterior",
    "prior",
    "experimental design",
    "value of information",
    "bandit",
    "arms",
]


def _synthetic_inputs():
    surface = {
        "alpha.medium": {"python", "docs"},
        "beta.high": {"docs", "data"},
        "gamma.low": set(),
    }
    coverage = {
        "alpha.medium": {"verified": 2},
        "beta.high": {"heuristic": 2, "asserted": 1},
        "gamma.low": {},
    }
    return surface, coverage


class GapDetectionTests(unittest.TestCase):
    def test_detects_all_three_gap_types_with_correct_fields(self):
        surface, coverage = _synthetic_inputs()

        gaps = detect_gaps(surface, coverage, min_overlap_cells=2)

        self.assertEqual(
            gaps,
            [
                UnvisitedArm(arm="gamma.low"),
                ConfoundedPair(
                    arm_a="alpha.medium",
                    arm_b="beta.high",
                    shared_cells=1,
                    min_overlap_cells=2,
                ),
                ObserverLimited(
                    arm="beta.high",
                    tiers=("asserted", "heuristic"),
                    n_grades=3,
                ),
            ],
        )
        self.assertEqual(
            gaps[0].description,
            "Workflow configuration gamma.low has no observed runs.",
        )
        self.assertEqual(
            gaps[1].description,
            "Workflow configurations alpha.medium and beta.high share 1 task cell, "
            "below the required 2.",
        )
        self.assertEqual(
            gaps[2].description,
            "Workflow configuration beta.high has grades only from asserted, heuristic evidence.",
        )

    def test_descriptions_are_single_sentences_and_forbidden_vocabulary_safe(self):
        surface, coverage = _synthetic_inputs()

        for gap in detect_gaps(surface, coverage, min_overlap_cells=2):
            self.assertTrue(gap.description.endswith("."))
            self.assertNotIn(". ", gap.description[:-1])
            lowered = gap.description.lower()
            for term in FORBIDDEN:
                self.assertNotIn(term, lowered)
            self.assertNotIn("—", gap.description)

    def test_strong_grade_prevents_observer_limited_gap_and_objects_are_immutable(self):
        gaps = detect_gaps(
            {"alpha": {"one"}, "beta": {"one"}},
            {"alpha": {"reported": 1, "heuristic": 4}, "beta": {"verified": 1}},
            min_overlap_cells=1,
        )

        self.assertEqual(gaps, [])
        gap = UnvisitedArm("unused")
        with self.assertRaises(FrozenInstanceError):
            gap.arm = "changed"

    def test_overlap_threshold_must_be_positive(self):
        for threshold in (0, -1):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "at least 1"):
                    detect_gaps({}, {}, min_overlap_cells=threshold)

    def test_invalid_tier_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            detect_gaps({"alpha": {"one"}}, {"alpha": {"verified": -1}})


if __name__ == "__main__":
    unittest.main()
