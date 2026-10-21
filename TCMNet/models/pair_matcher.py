"""Shared pairwise coarse-to-fine matcher."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from cycle.tcm_fine import TCMFineMatching, TCMFinePreprocess

from .backbone import build_backbone
from .tcm_coarse import TCMCoarse

INF = 1e9


def mask_border(m, b: int, v):
    """Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        # -- # for training the fine-level pair matcher
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

        self.temperature = config["dsmax_temperature"]

        # optional TCMCoarse coarse-similarity refinement (disabled by default;
        # adds no parameters to the state_dict unless tcm_coarse_enable=True
        # keys are present, but the module object is always constructed so the
        # forward path stays uniform).
        self.tcm_coarse = TCMCoarse(config)

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)

        # optional learnable neighborhood-consensus re-scoring (residual,
        # applied before the softmaxes so both directions and the coarse loss
        # see the refined scores). A zero delta / disabled module leaves the
        # path byte-identical to the pretrained pairwise matcher.
        if self.tcm_coarse.enable:
            delta = self.tcm_coarse(sim_matrix, data, mask_c0, mask_c1)
            if delta is not None:
                sim_matrix = sim_matrix + delta

        conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
        conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
        data.update(
            {"conf_matrix_0_to_1": conf_matrix_0_to_1, "conf_matrix_1_to_0": conf_matrix_1_to_0}
        )

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        """
        Args:
            conf_matrix_0_to_1 (torch.Tensor): [N, L, S]
            conf_matrix_1_to_0 (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        _device = conf_matrix_0_to_1.device

        # 1. confidence thresholding
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > self.thr)
            * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > self.thr)
            * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]),
        )

        mask = rearrange(mask, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", **axes_lengths)
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
        mask = rearrange(mask, "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)", **axes_lengths)

        # 2. find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(
            conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids]
        )

        # 3. Random sampling of training samples for the fine-level matcher
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, (
                "min-num-gt-pad should be less than num-train-matches"
            )

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min,),
                    device=_device,
                )

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=_device,
            )
            mconf_gt = torch.zeros(
                len(data["spv_b_ids"]), device=_device
            )  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                ),
            )

        # These matches select patches that feed into fine-level network
        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = (
            torch.stack(
                [
                    i_ids % data["hw0_c"][1],
                    torch.div(i_ids, data["hw0_c"][1], rounding_mode="trunc"),
                ],
                dim=1,
            )
            * scale0
        )
        mkpts1_c = (
            torch.stack(
                [
                    j_ids % data["hw1_c"][1],
                    torch.div(j_ids, data["hw1_c"][1], rounding_mode="trunc"),
                ],
                dim=1,
            )
            * scale1
        )

        # These matches is the current prediction (for visualization)
        coarse_matches.update(
            {
                "gt_mask": mconf == 0,
                "m_bids": b_ids[mconf != 0],  # mconf == 0 => gt matches
                "mkpts0_c": mkpts0_c[mconf != 0],
                "mkpts1_c": mkpts1_c[mconf != 0],
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches


class PairMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone()
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = TCMFinePreprocess(config)
        self.fine_matching = TCMFineMatching(config)

    def extract_pair_features(self, image0, image1):
        """Extract features jointly for one image pair.

        The paired window-attention backbone couples the two batch halves, so
        features cannot be extracted from an isolated image with this
        backbone. Pair inputs must currently have the same spatial shape.
        """
        if image0.shape[0] != image1.shape[0]:
            raise ValueError("Both images must have the same batch size.")
        if image0.shape[2:] != image1.shape[2:]:
            raise ValueError("Paired hierarchical feature extraction requires equal image shapes.")
        batch_size = image0.shape[0]
        feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
        feat_c0, feat_c1 = feats_c.split(batch_size)
        feat_f0, feat_f1 = feats_f.split(batch_size)
        return (feat_c0, feat_f0), (feat_c1, feat_f1)

    def forward_coarse(self, data, features0=None, features1=None):
        """Run feature extraction (if needed) and coarse matching.

        The returned state contains the feature tensors required by
        :meth:`forward_fine`. Supplying precomputed features lets a tri-modal
        caller reuse features already extracted for that same image pair.
        """
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if (features0 is None) != (features1 is None):
            raise ValueError("features0 and features1 must be supplied together.")

        if features0 is None:
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.extract_pair_features(
                data["image0"], data["image1"]
            )
        else:
            feat_c0, feat_f0 = features0
            feat_c1, feat_f1 = features1

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # Flatten coarse feature maps to [N, HW, C].
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        return {
            "data": data,
            "feat_c0": feat_c0,
            "feat_c1": feat_c1,
            "feat_f0": feat_f0,
            "feat_f1": feat_f1,
        }

    def forward_fine(self, state, selected_indices=None, anchor_ids=None):
        """Run fine matching, optionally on a filtered subset of coarse rows.

        ``selected_indices`` indexes the coarse matches produced by
        :meth:`forward_coarse`. ``anchor_ids`` assigns a shared tri-cycle ID to
        every selected row so results from all three pair directions can be
        joined after fine matching.
        """
        data = state["data"]
        device = data["b_ids"].device

        if selected_indices is not None:
            if self.training:
                raise RuntimeError("Filtered fine matching is an inference-only API.")
            selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=device)
            if selected_indices.ndim != 1:
                raise ValueError("selected_indices must be one-dimensional.")
            if selected_indices.numel() > 0:
                if selected_indices.min() < 0 or selected_indices.max() >= len(data["b_ids"]):
                    raise IndexError("selected_indices contains an invalid coarse row.")

            for key in (
                "b_ids",
                "i_ids",
                "j_ids",
                "gt_mask",
                "m_bids",
                "mkpts0_c",
                "mkpts1_c",
                "mconf",
            ):
                if key in data:
                    data[key] = data[key][selected_indices]

        coarse_count = len(data["b_ids"])
        if anchor_ids is None:
            anchor_ids = torch.arange(coarse_count, dtype=torch.long, device=device)
        else:
            anchor_ids = torch.as_tensor(anchor_ids, dtype=torch.long, device=device)
            if anchor_ids.ndim != 1 or len(anchor_ids) != coarse_count:
                raise ValueError("anchor_ids must have one value per selected coarse row.")
        data["coarse_anchor_ids"] = anchor_ids

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            state["feat_f0"],
            state["feat_f1"],
            state["feat_c0"],
            state["feat_c1"],
            data,
        )

        self.fine_matching(
            feat_f0_unfold,
            feat_f1_unfold,
            state["feat_f0"],
            state["feat_f1"],
            data,
        )
        return data

    def forward(self, data):
        """Run the original pairwise coarse-to-fine inference/training path."""
        state = self.forward_coarse(data)
        self.forward_fine(state)

    def load_state_dict(self, state_dict, *args, **kwargs):
        from .checkpoint import normalize_checkpoint

        state_dict = normalize_checkpoint(state_dict)
        # Preserve the caller's checkpoint while removing Lightning prefixes.
        state_dict = {
            (key[len("matcher.") :] if key.startswith("matcher.") else key): value
            for key, value in state_dict.items()
        }
        return super().load_state_dict(state_dict, *args, **kwargs)
