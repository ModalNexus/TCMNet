"""Behavioral checks for support routing and mandatory output verification."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from cycle.refinement import _augment_topk, _node_aware_one_to_one
from cycle.tcm_fine import TCMFine, _geometric_filter
from infer_tcmnet import parse_args
from models.derived_feature_verifier import _direction_test
from models.tcmnet import TCMNet


def empty():
    return np.empty((0, 5), dtype=np.float64)


def grid_matches(shift=(7.0, 11.0)):
    points = np.array([(30.0 * x, 30.0 * y) for y in range(4) for x in range(5)])
    return np.column_stack((points, points + shift, np.full(len(points), 0.99)))


class SupportRoutingTests(unittest.TestCase):
    def test_two_hop_completion_tries_verified_alternative_before_occupying_source(self):
        def inputs():
            return {
                "rs": empty(),
                "si": np.array([
                    [20, 20, 30, 30, 0.99],
                    [40, 40, 30, 30, 0.80],
                ]),
                "ir": np.array([[30, 30, 10, 10, 0.99]]),
            }

        options = dict(
            epsilon=5, augment_radius=3, min_edge_conf=0,
            link_sigma=None, check_target=False, local_verify=False,
            local_k=12, local_thresh=4, replace_direct=False,
            replace_margin=0, topk_k=3, topk_lambda_geo=0.1,
        )
        legacy, _ = _augment_topk(enhanced=inputs(), **options)
        np.testing.assert_allclose(legacy["rs"][:, 2:4], [[20, 20]])

        def verify(pair, rows):
            passed = np.isclose(rows[:, 2], 40) if pair == "rs" else np.ones(len(rows), bool)
            return passed, np.zeros(len(rows)), np.zeros(len(rows))

        checked, mask = _augment_topk(
            enhanced=inputs(), path_verifier=verify, **options
        )
        np.testing.assert_allclose(checked["rs"][:, 2:4], [[40, 40]])
        self.assertEqual(mask["rs"].tolist(), [True])

    def test_tcm_fine_owns_the_transition_from_coarse_to_fine(self):
        class StubMatcher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coarse_calls = 0
                self.fine_calls = 0
                self.pairs = []

            def forward_coarse(self, data):
                self.coarse_calls += 1
                self.pairs.append((data["image0"].item(), data["image1"].item()))
                return {"data": data}

            def forward_fine(self, state):
                self.fine_calls += 1
                return {
                    "mkpts0_f": torch.tensor([[10.0, 10.0]]),
                    "mkpts1_f": torch.tensor([[20.0, 20.0]]),
                    "mconf_f": torch.tensor([0.9]),
                }

        matcher = StubMatcher()
        network = TCMNet(
            matcher, verify_derived=False, refine_cycles=False,
            do_filter=False, do_correction=False, do_augmentation=False,
        )
        images = [torch.full((1, 1, 1, 1), value) for value in (1.0, 2.0, 3.0)]
        result = network(modality1=images[0], modality2=images[1], modality3=images[2])
        self.assertEqual((matcher.coarse_calls, matcher.fine_calls), (3, 3))
        self.assertEqual(matcher.pairs, [(1.0, 2.0), (2.0, 3.0), (3.0, 1.0)])
        for pair in ("rs", "si", "ir"):
            np.testing.assert_allclose(result["baseline"][pair], [[10, 10, 20, 20, 0.9]])

    def test_conflicting_triangles_cannot_reuse_the_third_edge(self):
        cycles = [
            dict(rs_index=0, si_index=0, ir_index=0, score=-1),
            dict(rs_index=1, si_index=1, ir_index=0, score=-2),
        ]
        node_info = {
            "edge_nodes": {
                "rs": np.array([[100, 200, 0], [101, 201, 1]]),
                "si": np.array([[200, 300, 0], [201, 301, 1]]),
                "ir": np.array([[300, 100, 0]]),
            }
        }
        self.assertEqual(_node_aware_one_to_one(cycles, node_info), [cycles[0]])

    def test_isolated_open_and_closed_evidence_are_distinct(self):
        rs = np.array(
            [[1000, 1000, 2000, 2000, 0.99], [10, 10, 20, 20, 0.99], [100, 100, 200, 200, 0.99]]
        )
        si = np.array([[20.5, 20, 30, 30, 0.99], [200.2, 200, 300, 300, 0.99]])
        ir = np.array([[300.2, 300, 100.2, 100, 0.99]])
        result = TCMFine(do_filter=False, epsilon=2, tau=2)(rs, si, ir)
        np.testing.assert_array_equal(result["baseline_support"]["rs"], [1, 2, 3])
        np.testing.assert_array_equal(result["baseline_support"]["si"], [2, 3])
        np.testing.assert_array_equal(result["baseline_support"]["ir"], [3])
        np.testing.assert_array_equal(result["enhanced"]["rs"][0], rs[0])
        self.assertEqual(len(result["cycles"]), 1)
        new_ir = result["enhanced"]["ir"][result["derived_mask"]["ir"]]
        np.testing.assert_allclose(new_ir[:, :4], [[30, 30, 10, 10]])
        np.testing.assert_array_equal(
            result["edge_support"]["ir"][result["derived_mask"]["ir"]], [2]
        )

    def test_single_edges_use_shared_direction_geometry(self):
        matches = grid_matches()
        result = TCMFine()(matches, empty(), empty())
        np.testing.assert_allclose(result["enhanced"]["rs"], matches)
        np.testing.assert_array_equal(result["baseline_support"]["rs"], np.ones(len(matches)))
        self.assertEqual(result["geometry"]["rs"]["status"], "ok")
        self.assertEqual(result["stats"]["n_cycles"], 0)

    def test_two_edges_complete_without_creating_independent_cycles(self):
        rs = grid_matches((10, 10))
        si = rs.copy()
        si[:, :4] += 10
        result = TCMFine()(rs, si, empty())
        self.assertEqual(result["stats"]["n_cycles"], 0)
        self.assertEqual(result["stats"]["n_output_cycles"], 0)
        self.assertEqual(len(result["enhanced"]["ir"]), len(rs))
        np.testing.assert_allclose(result["enhanced"]["ir"][:, :2], si[:, 2:4])
        np.testing.assert_allclose(result["enhanced"]["ir"][:, 2:4], rs[:, :2])
        self.assertTrue(result["derived_mask"]["ir"].all())
        self.assertTrue((result["edge_support"]["ir"] == 2).all())


class GeometryVerificationTests(unittest.TestCase):
    def test_local_feature_peak_must_be_near_the_proposed_match(self):
        query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        candidates = torch.zeros(2, 25, 2)
        candidates[:, :, 1] = 1.0
        candidates[0, 12] = torch.tensor([1.0, 0.0])
        candidates[1, 14] = torch.tensor([1.0, 0.0])
        accepted, peak = _direction_test(
            query, candidates, torch.ones(2, 25, dtype=torch.bool), 0.1, 0.12, 1.0
        )
        self.assertEqual(accepted.tolist(), [True, False])
        self.assertTrue((peak > 0.8).all())

    def test_feature_gate_only_removes_borderline_derived_rows(self):
        rows = grid_matches()
        rs = rows.copy()
        si = rows.copy()
        si[:, :4] += 11
        ordinary = TCMFine()(rs, si, empty())
        self.assertEqual(len(ordinary["enhanced"]["ir"]), len(rows))

        def reject_every_feature(result):
            result["feature_verified_mask"] = {}
            result["derived_feature_verification"] = {}
            for pair, matched in result["enhanced"].items():
                result["feature_verified_mask"][pair] = ~result["derived_mask"][pair]
                n = int(result["derived_mask"][pair].sum())
                result["derived_feature_verification"][pair] = {
                    "candidates": n,
                    "accepted": 0,
                    "rejected": n,
                }
            return result

        checked = TCMFine()(rs, si, empty(), derived_verifier=reject_every_feature)
        np.testing.assert_array_equal(checked["enhanced"]["rs"], ordinary["enhanced"]["rs"])
        np.testing.assert_array_equal(checked["enhanced"]["si"], ordinary["enhanced"]["si"])
        # All grid completions have zero fitted residual and remain inside the
        # safe geometric band even when the feature callback rejects them.
        np.testing.assert_array_equal(checked["enhanced"]["ir"], ordinary["enhanced"]["ir"])
        self.assertEqual(checked["stats"]["feature_rejected_ir"], 0)
        no_geometry = TCMFine(do_filter=False)(
            rs, si, empty(), derived_verifier=reject_every_feature
        )
        self.assertEqual(len(no_geometry["enhanced"]["rs"]), len(rs))
        self.assertEqual(len(no_geometry["enhanced"]["si"]), len(si))
        self.assertEqual(len(no_geometry["enhanced"]["ir"]), 0)
        self.assertEqual(no_geometry["stats"]["feature_rejected_ir"], len(rows))

    def test_cycle_outlier_is_removed_and_output_indices_are_remapped(self):
        rows = grid_matches()
        baseline = {pair: rows.copy() for pair in ("rs", "si", "ir")}
        baseline["rs"][0, 2:4] += 100
        result = {
            "baseline": baseline,
            "enhanced": {pair: value.copy() for pair, value in baseline.items()},
            "derived_mask": {pair: np.zeros(len(rows), dtype=bool) for pair in baseline},
            "cycles": [dict(rs_index=i, si_index=i, ir_index=i) for i in (0, 5)],
            "stats": {},
        }
        result["derived_mask"]["rs"][-1] = True
        output = _geometric_filter(result, 3.0)
        self.assertFalse(output["filter_masks"]["rs"][0])
        self.assertEqual(output["geometry"]["rs"]["cycle_outliers"], 1)
        self.assertEqual(output["geometry"]["rs"]["protected_cycle_outliers"], 0)
        self.assertEqual(len(output["output_cycles"]), 1)
        cycle = output["output_cycles"][0]
        self.assertEqual(cycle["rs_index"], 4)
        self.assertEqual(cycle["baseline_rs_index"], 5)
        np.testing.assert_array_equal(output["enhanced"]["rs"][4], baseline["rs"][5])
        self.assertTrue(output["derived_mask"]["rs"][-1])
        self.assertEqual(output["stats"]["enhanced_rs"], len(rows) - 1)
        self.assertEqual(output["stats"]["prefilter_rs"], len(rows))
        for pair, kept in output["enhanced"].items():
            H = output["geometry"][pair]["homography"]
            projected = np.column_stack((kept[:, :2], np.ones(len(kept)))) @ H.T
            errors = np.linalg.norm(projected[:, :2] / projected[:, 2:] - kept[:, 2:4], axis=1)
            self.assertTrue((errors <= 3).all())
        first_output = output["enhanced"]["rs"].copy()
        np.testing.assert_array_equal(_geometric_filter(output, 3)["enhanced"]["rs"], first_output)

    def test_fewer_than_four_or_collinear_points_have_diagnostics(self):
        for rows, status in (
            (grid_matches()[:3], "insufficient_matches"),
            (grid_matches()[:5], "degenerate_points"),
        ):
            result = TCMFine()(rows, empty(), empty())
            self.assertEqual(len(result["enhanced"]["rs"]), 0)
            self.assertEqual(result["geometry"]["rs"]["status"], status)
            np.testing.assert_array_equal(result["prefilter"]["rs"], rows)

    def test_failed_estimation_does_not_emit_unverified_matches(self):
        with patch("cv2.findHomography", return_value=(None, None)):
            result = TCMFine()(grid_matches(), empty(), empty())
        self.assertEqual(result["geometry"]["rs"]["status"], "estimation_failed")
        self.assertEqual(len(result["enhanced"]["rs"]), 0)

    def test_invalid_geometry_threshold(self):
        for threshold in (-1, 0, float("nan")):
            with self.assertRaises(ValueError):
                TCMFine(filter_thresh=threshold)(empty(), empty(), empty())

    def test_cli_filter_is_enabled_by_default(self):
        args = ["--modality1", "r.png", "--modality2", "s.png", "--modality3", "i.png"]
        self.assertTrue(parse_args(args).filter)
        self.assertTrue(parse_args(args + ["--filter"]).filter)
        self.assertFalse(parse_args(args + ["--no-filter"]).filter)


if __name__ == "__main__":
    unittest.main()
