"""Behavioral checks for cycle-local proposals and atomic rollback."""

import unittest

import numpy as np

from models.cycle_feature_refiner import CycleFeatureRefiner


def triangle():
    return {
        "rs": np.array([[0.0, 0.0, 10.0, 0.0, 0.9]]),
        "si": np.array([[10.0, 0.0, 20.0, 0.0, 0.9]]),
        "ir": np.array([[20.0, 0.0, 0.0, 0.0, 0.9]]),
    }


class StubRefiner(CycleFeatureRefiner):
    def __init__(self, *, old, current, proposed, usable, target):
        self.rollback_margin = 2.0
        self.max_local_move = 1.0
        self.max_from_original = 4.0
        self.old = old
        self.current = current
        self.proposed = proposed
        self.usable = usable
        self.target = target

    def _score_direction(self, pair, corrected, original):
        target = corrected[:, 2:4].copy()
        if pair == "rs":
            target[0] = self.target
        return (
            np.array([self.old[pair]]),
            np.array([self.current[pair]]),
            np.array([self.proposed[pair]]),
            target,
            np.array([self.usable[pair]]),
        )


class CycleFeatureRefinerTests(unittest.TestCase):
    def setUp(self):
        self.original = triangle()
        self.corrected = {pair: rows.copy() for pair, rows in self.original.items()}
        self.corrected["rs"][0, 2] = 10.5
        self.cycle = {"rs_index": 0, "si_index": 0, "ir_index": 0, "correction_weight": 0.5}

    def test_rolls_back_all_three_edges_when_original_feature_support_wins(self):
        refiner = StubRefiner(
            old=dict.fromkeys(triangle(), 3.0),
            current=dict.fromkeys(triangle(), 1.0),
            proposed=dict.fromkeys(triangle(), 1.0),
            usable=dict.fromkeys(triangle(), False),
            target=np.array([10.0, 0.0]),
        )
        output, stats = refiner(self.corrected, self.original, [self.cycle], epsilon=5, tau=8)
        for pair in self.original:
            np.testing.assert_array_equal(output[pair], self.original[pair])
        self.assertEqual(stats["rolled_back"], 1)
        self.assertEqual(self.cycle["feature_action"], "rollback")

    def test_accepts_feature_gain_that_reduces_cycle_residual(self):
        refiner = StubRefiner(
            old=dict.fromkeys(triangle(), 1.0),
            current=dict.fromkeys(triangle(), 2.0),
            proposed={"rs": 3.0, "si": 2.0, "ir": 2.0},
            usable={"rs": True, "si": False, "ir": False},
            target=np.array([10.1, 0.0]),
        )
        output, stats = refiner(self.corrected, self.original, [self.cycle], epsilon=5, tau=8)
        np.testing.assert_allclose(output["rs"][0, 2:4], [10.1, 0.0])
        np.testing.assert_array_equal(output["si"], self.original["si"])
        self.assertEqual(stats["locally_refined"], 1)
        self.assertEqual(stats["edges_refined"], 1)

    def test_rejects_large_move_and_preserves_previous_correction(self):
        refiner = StubRefiner(
            old=dict.fromkeys(triangle(), 1.0),
            current=dict.fromkeys(triangle(), 2.0),
            proposed={"rs": 5.0, "si": 2.0, "ir": 2.0},
            usable={"rs": True, "si": False, "ir": False},
            target=np.array([12.0, 0.0]),
        )
        output, stats = refiner(self.corrected, self.original, [self.cycle], epsilon=5, tau=8)
        np.testing.assert_allclose(output["rs"][0, 2:4], [10.5, 0.0])
        self.assertEqual(stats["unchanged"], 1)
        self.assertEqual(self.cycle["feature_action"], "keep_correction")


if __name__ == "__main__":
    unittest.main()
