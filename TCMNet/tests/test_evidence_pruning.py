"""Cycle-branch and two-edge evidence pruning with explicit conflicts."""

import unittest

import numpy as np

from cycle.refinement import _augment_topk
from cycle.tcm_fine import TCMFine


def rows(*items):
    return np.asarray(items, dtype=np.float64).reshape(-1, 5)


def feature_evidence(_pair, matches):
    passed = matches[:, 4] >= 0.5
    peaks = np.where(passed, 0.9, 0.05)
    return passed, peaks, peaks


class EvidencePruningTests(unittest.TestCase):
    def matcher(self, **kwargs):
        return TCMFine(
            do_filter=False, do_correction=False, do_augmentation=False, **kwargs
        )

    def test_cycle_removes_incompatible_branches_and_preserves_single(self):
        rs = rows(
            (10, 10, 50, 20, 0.1),  # failed source conflict, before the cycle row
            (10, 10, 20, 20, 0.9),
            (70, 10, 20, 20, 0.1),  # failed target conflict
            (10, 10, 20.2, 20, 0.1),  # compatible duplicate, even if feature-failed
            (120, 110, 130, 120, 0.1),  # independent single edge
        )
        si = rows((20, 20, 30, 30, 0.9), (20, 20, 80, 30, 0.1))
        ir = rows((30, 30, 10, 10, 0.9), (30, 30, 90, 10, 0.1))
        result = self.matcher()(rs, si, ir, edge_verifier=feature_evidence)
        np.testing.assert_array_equal(result["deleted_mask"]["rs"], [True, False, True, False, False])
        np.testing.assert_array_equal(result["deleted_mask"]["si"], [False, True])
        np.testing.assert_array_equal(result["deleted_mask"]["ir"], [False, True])
        self.assertEqual(result["stats"]["protected_cycles"], 1)
        self.assertEqual(result["cycles"][0]["rs_index"], 1)
        self.assertEqual(result["working_cycles"][0]["rs_index"], 0)
        np.testing.assert_array_equal(result["post_prune_support"]["rs"], [3, 2, 1])
        self.assertEqual(result["deletion_reasons"]["rs"][0]["reason"], "protected_cycle_conflict")

    def test_competing_two_edge_paths_keep_verified_winner_and_shared_edge(self):
        rs = rows((10, 10, 20, 20, 0.9), (120, 110, 130, 120, 0.05))
        si = rows((20, 20, 30, 30, 0.9), (20, 20, 60, 60, 0.1))
        ir = rows()
        result = self.matcher()(rs, si, ir, edge_verifier=feature_evidence)
        np.testing.assert_array_equal(result["deleted_mask"]["rs"], [False, False])
        np.testing.assert_array_equal(result["deleted_mask"]["si"], [False, True])
        np.testing.assert_array_equal(result["post_prune_support"]["rs"], [2, 1])
        np.testing.assert_array_equal(result["post_prune_support"]["si"], [2])
        self.assertEqual(result["stats"]["selected_paths"], 1)
        self.assertEqual(result["stats"]["deleted_open_si"], 1)
        self.assertEqual(result["deletion_reasons"]["si"][0]["reason"], "competing_two_edge_branch")
        np.testing.assert_array_equal(result["enhanced"]["rs"][1], rs[1])

    def test_two_verified_incompatible_paths_keep_higher_score(self):
        rs = rows((10, 10, 20, 20, 0.9))
        si = rows((20, 20, 30, 30, 0.9), (20, 20, 60, 60, 0.7))
        result = self.matcher()(rs, si, rows(), edge_verifier=feature_evidence)
        self.assertEqual(result["stats"]["open_candidates"], 2)
        self.assertEqual(result["stats"]["deleted_total"], 1)
        np.testing.assert_array_equal(result["deleted_mask"]["si"], [False, True])
        np.testing.assert_array_equal(result["enhanced"]["si"], si[:1])

    def test_unverified_competition_keeps_original_edges(self):
        rs = rows((10, 10, 20, 20, 0.9))
        si = rows((20, 20, 30, 30, 0.9), (20, 20, 60, 60, 0.8))

        def one_direction_supported(_pair, matches):
            return (
                np.zeros(len(matches), dtype=bool),
                np.full(len(matches), 0.9),
                np.full(len(matches), 0.05),
            )

        preserved = self.matcher()(rs, si, rows(), edge_verifier=one_direction_supported)
        self.assertEqual(preserved["stats"]["deleted_total"], 0)
        np.testing.assert_array_equal(preserved["post_prune_support"]["si"], [2, 2])

    def test_best_verified_path_keeps_shared_edge_and_supplies_two_hop_completion(self):
        rs = rows((10, 10, 20, 20, 0.9))
        si = rows((20, 20, 30, 30, 0.95), (20, 20, 60, 60, 0.7))

        def verifier(pair, matches):
            passed = np.ones(len(matches), dtype=bool)
            if pair == "si":
                passed = matches[:, 4] < 0.9
            peaks = np.where(passed, 0.9, 0.05)
            return passed, peaks, peaks

        result = TCMFine(
            do_filter=False, do_correction=False, do_augmentation=True
        )(rs, si, rows(), edge_verifier=verifier)
        np.testing.assert_array_equal(result["deleted_mask"]["rs"], [False])
        np.testing.assert_array_equal(result["deleted_mask"]["si"], [True, False])
        np.testing.assert_array_equal(result["post_prune_support"]["rs"], [2])
        self.assertGreater(len(result["enhanced"]["ir"]), 0)

    def test_cycle_conflict_deletes_even_without_feature_verifier(self):
        rs = rows((10, 10, 20, 20, 0.9), (10, 10, 50, 20, 0.8))
        si = rows((20, 20, 30, 30, 0.9))
        ir = rows((30, 30, 10, 10, 0.9))

        def failing_verifier(_pair, matches):
            return (
                np.zeros(len(matches), dtype=bool),
                np.full(len(matches), 0.05),
                np.full(len(matches), 0.05),
            )

        result = self.matcher()(rs, si, ir, edge_verifier=failing_verifier)
        self.assertEqual(result["stats"]["protected_cycles"], 1)
        self.assertEqual(result["stats"]["deleted_rs"], 1)

        no_verifier = self.matcher()(rs, si, ir)
        self.assertEqual(no_verifier["stats"]["deleted_total"], 1)

        disabled = self.matcher(do_evidence_pruning=False)(
            rs, si, ir, edge_verifier=feature_evidence
        )
        self.assertEqual(disabled["stats"]["deleted_rs"], 0)

    def test_cycle_exterior_uses_exact_one_px_instead_of_qs_node_cluster(self):
        rs = rows(
            (10, 10, 20, 20, 0.9),
            (11, 10, 60, 20, 0.1),
            (11.1, 10, 70, 20, 0.1),
        )
        si = rows((20, 20, 30, 30, 0.9))
        ir = rows((30, 30, 10, 10, 0.9))
        result = self.matcher(epsilon=5)(rs, si, ir, edge_verifier=feature_evidence)
        np.testing.assert_array_equal(result["deleted_mask"]["rs"], [False, True, False])
        self.assertEqual(result["stats"]["protected_cycles"], 1)

    def test_cycle_exterior_needs_divergence_but_not_feature_failure(self):
        rs = rows(
            (10, 10, 20, 20, 0.9),
            (10, 10, 23, 20, 0.1),   # 3 px: not divergent
            (10, 10, 24, 20, 0.1),   # exactly 4 px: not divergent
            (10, 10, 60, 20, 0.8),   # divergent and feature-verified: still deleted
        )
        si = rows((20, 20, 30, 30, 0.9))
        ir = rows((30, 30, 10, 10, 0.9))
        result = self.matcher()(rs, si, ir, edge_verifier=feature_evidence)
        np.testing.assert_array_equal(result["deleted_mask"]["rs"], [False, False, False, True])

    def test_competing_paths_use_exact_one_px_not_qs_node_cluster(self):
        si = rows((20, 20, 30, 30, 0.9), (60, 20, 70, 60, 0.8))
        rs_far = rows((10, 10, 20, 20, 0.9), (11.1, 10, 60, 20, 0.1))
        far = self.matcher(epsilon=5)(rs_far, si, rows(), edge_verifier=feature_evidence)
        self.assertEqual(far["stats"]["open_candidates"], 0)
        self.assertEqual(far["stats"]["deleted_total"], 0)
        np.testing.assert_array_equal(far["post_prune_support"]["rs"], [2, 2])

        rs_near = rows((10, 10, 20, 20, 0.9), (11, 10, 60, 20, 0.1))
        near = self.matcher(epsilon=5)(rs_near, si, rows(), edge_verifier=feature_evidence)
        self.assertEqual(near["stats"]["open_candidates"], 2)
        np.testing.assert_array_equal(near["deleted_mask"]["rs"], [False, True])
        np.testing.assert_array_equal(near["deleted_mask"]["si"], [False, True])

    def test_competing_paths_require_other_endpoint_strictly_over_four_px(self):
        def run(second_target):
            rs = rows((10, 10, 20, 20, 0.9), (10, 10, second_target, 20, 0.8))
            si = rows((20, 20, 30, 30, 0.9), (second_target, 20, 60, 60, 0.8))
            return self.matcher()(rs, si, rows(), edge_verifier=feature_evidence)

        for target in (23, 24):
            boundary = run(target)
            self.assertEqual(boundary["stats"]["open_candidates"], 0)
            self.assertEqual(boundary["stats"]["deleted_total"], 0)

        divergent = run(24.1)
        self.assertEqual(divergent["stats"]["open_candidates"], 2)
        np.testing.assert_array_equal(divergent["deleted_mask"]["rs"], [False, True])
        np.testing.assert_array_equal(divergent["deleted_mask"]["si"], [False, True])

    def test_two_edge_candidate_link_uses_exact_half_px(self):
        rs = rows((10, 10, 20, 20, 0.9))
        si_near = rows((20, 20, 30, 30, 0.9), (20.5, 20, 60, 60, 0.1))
        si_far = rows((20, 20, 30, 30, 0.9), (20.6, 20, 60, 60, 0.1))

        near = self.matcher(epsilon=5)(rs, si_near, rows(), edge_verifier=feature_evidence)
        np.testing.assert_array_equal(near["deleted_mask"]["si"], [False, True])

        far = self.matcher(epsilon=5)(rs, si_far, rows(), edge_verifier=feature_evidence)
        self.assertEqual(far["stats"]["deleted_total"], 0)
        np.testing.assert_array_equal(far["post_prune_support"]["si"], [2, 2])

    def test_pruning_radius_does_not_change_existing_two_hop_radius(self):
        rs = rows((10, 10, 20, 20, 0.9))
        si = rows((21.4, 20, 30, 30, 0.9))
        result = TCMFine(
            epsilon=5, do_filter=False, do_correction=False, do_augmentation=True
        )(rs, si, rows(), edge_verifier=feature_evidence)
        self.assertEqual(result["stats"]["deleted_total"], 0)
        np.testing.assert_array_equal(result["post_prune_support"]["rs"], [2])
        np.testing.assert_array_equal(result["post_prune_support"]["si"], [2])
        self.assertGreater(len(result["enhanced"]["ir"]), 0)

    def test_noncompeting_single_and_two_edge_paths_are_preserved(self):
        rs = rows((10, 10, 20, 20, 0.05), (100, 100, 110, 110, 0.05))
        si = rows((20, 20, 30, 30, 0.9))
        ir = rows()
        result = self.matcher()(rs, si, ir, edge_verifier=feature_evidence)
        self.assertEqual(result["stats"]["open_candidates"], 0)
        self.assertEqual(result["stats"]["deleted_total"], 0)
        np.testing.assert_array_equal(result["post_prune_support"]["rs"], [2, 1])
        np.testing.assert_array_equal(result["enhanced"]["rs"], rs)

    def test_geometry_filter_uses_remapped_cycle_indices(self):
        rgb = np.array([(30.0 * x, 30.0 * y) for y in range(3) for x in range(3)])
        sar, infrared = rgb + (10, 0), rgb + (20, 0)
        rs = np.column_stack((rgb, sar, np.full(len(rgb), 0.9)))
        si = np.column_stack((sar, infrared, np.full(len(rgb), 0.9)))
        ir = np.column_stack((infrared, rgb, np.full(len(rgb), 0.9)))
        rs = np.vstack(([[0, 0, 80, 0, 0.1]], rs))
        result = TCMFine(do_filter=True, do_correction=False, do_augmentation=False)(
            rs, si, ir, edge_verifier=feature_evidence
        )
        self.assertEqual(result["stats"]["deleted_conflict_rs"], 1)
        self.assertEqual(result["stats"]["n_output_cycles"], len(rgb))
        self.assertTrue(all(cycle["rs_index"] < len(rgb) for cycle in result["output_cycles"]))

    def test_deleted_edge_cannot_be_synthesized_again_and_no_duplicate_source(self):
        empty = rows()
        rs = rows((10, 10, 20, 20, 0.1))
        si = rows((20, 20, 30, 30, 0.9))
        ir = rows((30, 30, 10, 10, 0.9), (30, 30, 10, 10, 0.8))
        stats = {"blocked_backfills": 0}
        enhanced, _ = _augment_topk(
            {"rs": empty.copy(), "si": si.copy(), "ir": ir.copy()},
            epsilon=2, augment_radius=3, min_edge_conf=0, link_sigma=None,
            check_target=False, local_verify=False, local_k=12, local_thresh=4,
            replace_direct=False, replace_margin=0, topk_k=3, topk_lambda_geo=0.1,
            excluded_edges={"rs": rs, "si": empty, "ir": empty},
            exclude_radius=3, augmentation_stats=stats,
        )
        self.assertEqual(len(enhanced["rs"]), 0)
        self.assertEqual(stats["blocked_backfills"], 2)

        enhanced, _ = _augment_topk(
            {"rs": empty.copy(), "si": si.copy(), "ir": ir.copy()},
            epsilon=2, augment_radius=3, min_edge_conf=0, link_sigma=None,
            check_target=False, local_verify=False, local_k=12, local_thresh=4,
            replace_direct=False, replace_margin=0, topk_k=3, topk_lambda_geo=0.1,
        )
        self.assertEqual(len(enhanced["rs"]), 1)


if __name__ == "__main__":
    unittest.main()
