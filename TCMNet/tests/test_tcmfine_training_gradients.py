"""Regression checks for the two trainable TCMFine losses."""

import unittest

import torch

from configs.default import get_cfg_defaults, get_matcher_config
from cycle.tcm_fine import TCMFineMatching, TCMFinePreprocess
from models.tcm_coarse import TCMCoarse
from training.loss import TCMFineLoss
from training.supervision import compute_supervision_coarse


class TCMFineGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = get_matcher_config()

    @staticmethod
    def _data():
        return {
            "b_ids": torch.tensor([0]),
            "i_ids": torch.tensor([5]),
            "j_ids": torch.tensor([5]),
            "hw0_c": (4, 4),
            "hw1_c": (4, 4),
            "hw0_f": (16, 16),
            "hw1_f": (16, 16),
            "hw0_i": (32, 32),
            "hw1_i": (32, 32),
            "coarse_anchor_ids": torch.tensor([0]),
            "T_0to1": torch.tensor([[[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]]),
        }

    def test_no_groundtruth_coarse_match_uses_pair_id_fallback(self):
        transform = torch.eye(3).unsqueeze(0)
        transform[:, 0, 2] = 10000.0
        data = {
            "image0": torch.zeros(1, 1, 32, 32),
            "image1": torch.zeros(1, 1, 32, 32),
            "T_0to1": transform,
            "T_1to0": torch.linalg.inv(transform),
            "pair_id": torch.tensor([7]),
        }
        compute_supervision_coarse(data, get_cfg_defaults())
        self.assertEqual(data["conf_matrix_gt"].count_nonzero().item(), 0)
        self.assertEqual(data["spv_b_ids"].tolist(), [0])
        self.assertEqual(data["spv_i_ids"].tolist(), [0])
        self.assertEqual(data["spv_j_ids"].tolist(), [0])

    def test_tcmfine_window_loss_reaches_feature_projection(self):
        preprocess = TCMFinePreprocess(self.config)
        matching = TCMFineMatching(self.config)
        data = self._data()
        feat0 = torch.randn(1, 96, 16, 16)
        feat1 = torch.randn(1, 96, 16, 16)
        coarse0 = torch.randn(1, 16, 192)
        coarse1 = torch.randn(1, 16, 192)
        windows = preprocess(feat0, feat1, coarse0, coarse1, data)
        matching(*windows, feat0, feat1, data)
        gt = torch.zeros_like(data["conf_matrix_fine"])
        gt[0, 12, 12] = 1
        data["conf_matrix_f_gt"] = gt
        loss_config = {"pair_matcher": self.config}
        loss = TCMFineLoss(loss_config).compute_tcmfine_window_loss(data)
        loss.backward()
        self.assertGreater(preprocess.merge_feat.weight.grad.abs().sum().item(), 0)

    def test_tcmfine_subpixel_loss_reaches_regression_head(self):
        matching = TCMFineMatching(self.config)
        data = self._data()
        confidence = torch.full((1, 25, 25), 0.001)
        confidence[0, 12, 12] = 0.95
        fine0 = torch.randn(1, 25, 96)
        fine1 = torch.randn(1, 25, 96)
        feature0 = torch.randn(1, 96, 16, 16)
        feature1 = torch.randn(1, 96, 16, 16)
        data.update(matching.get_fine_match(confidence, fine0, fine1, feature0, feature1, data))
        loss_config = {"pair_matcher": self.config}
        loss = TCMFineLoss(loss_config).compute_tcmfine_subpixel_loss(data)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertGreater(matching.subpixel_conv_block[0].weight.grad.abs().sum().item(), 0)

    def test_empty_threshold_fallback_preserves_probability_tensor(self):
        matching = TCMFineMatching(self.config)
        data = self._data()
        confidence = torch.full((1, 25, 25), 0.001, requires_grad=True)
        original = confidence.detach().clone()
        fine0 = torch.randn(1, 25, 96)
        fine1 = torch.randn(1, 25, 96)
        feature0 = torch.randn(1, 96, 16, 16)
        feature1 = torch.randn(1, 96, 16, 16)
        result = matching.get_fine_match(
            confidence, fine0, fine1, feature0, feature1, data
        )
        self.assertTrue(torch.equal(confidence.detach(), original))
        self.assertEqual(result["mconf_f"].item(), 1.0)

    def test_coarse_loss_reaches_residual_recalibration(self):
        config = dict(self.config["match_coarse"], tcm_coarse_enable=True)
        recalibration = TCMCoarse(config)
        similarity = torch.randn(1, 4, 4)
        data = {
            "hw0_c": (2, 2), "hw1_c": (2, 2),
            "spv_b_ids": torch.tensor([0]),
            "spv_i_ids": torch.tensor([1]),
            "spv_j_ids": torch.tensor([1]),
            "conf_matrix_gt": torch.zeros(1, 4, 4),
        }
        data["conf_matrix_gt"][0, 1, 1] = 1
        corrected = similarity + recalibration(similarity, data)
        data["conf_matrix_0_to_1"] = corrected.softmax(dim=2)
        data["conf_matrix_1_to_0"] = corrected.softmax(dim=1)
        loss = TCMFineLoss({"pair_matcher": self.config}).compute_coarse_loss(data)
        loss.backward()
        self.assertGreater(recalibration.conv[-1].convs[1].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
