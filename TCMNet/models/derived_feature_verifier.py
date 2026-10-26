"""Check synthesized edges with the frozen pair matcher's fine features."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _sample(feature_map, points, image_hw):
    """Bilinearly sample [N, C] descriptors at image-resolution xy coordinates."""
    height, width = image_hw
    feature_height, feature_width = feature_map.shape[-2:]
    xy = points.to(feature_map.dtype)
    grid = torch.stack(
        (
            xy[..., 0] * (feature_width / width) * (2 / (feature_width - 1)) - 1,
            xy[..., 1] * (feature_height / height) * (2 / (feature_height - 1)) - 1,
        ),
        dim=-1,
    )
    shape = grid.shape[:-1]
    sampled = F.grid_sample(
        feature_map, grid.reshape(1, -1, 1, 2), mode="bilinear", align_corners=True
    )
    return sampled[0, :, :, 0].T.reshape(*shape, feature_map.shape[1])


def _coarse_feature(coarse, points, coarse_hw, image_hw):
    """Nearest coarse anchor descriptor for each proposed image position."""
    coarse_height, coarse_width = coarse_hw
    image_height, image_width = image_hw
    index_x = (points[:, 0] * coarse_width / image_width).round().long().clamp(0, coarse_width - 1)
    index_y = (
        (points[:, 1] * coarse_height / image_height).round().long().clamp(0, coarse_height - 1)
    )
    return coarse[0, index_y * coarse_width + index_x]


def _pair_context(src, dst, coarse0, coarse1, data, preprocessing):
    """Project both anchors with the checkpoint's fine context weights."""
    if not preprocessing.cat_c_feat:
        return None, None
    c0 = _coarse_feature(coarse0, src, data["hw0_c"], data["hw0_i"])
    c1 = _coarse_feature(coarse1, dst, data["hw1_c"], data["hw1_i"])
    return preprocessing.down_proj(c0), preprocessing.down_proj(c1)


def _merge_context(features, context, preprocessing):
    if context is None:
        return features
    if features.ndim == 3:
        context = context[:, None, :].expand(-1, features.shape[1], -1)
    return preprocessing.merge_feat(torch.cat((features, context), dim=-1))


def _direction_test(query, candidates, valid, temperature, min_peak_probability, max_shift):
    """Require a distinct local peak near the proposed target position."""
    # Match FineMatching.forward: dot product after each descriptor is scaled
    # by sqrt(C), followed by the checkpoint's fine softmax temperature.
    q = query.float()
    c = candidates.float()
    similarity = (q[:, None, :] * c).sum(-1) / (q.shape[-1] * temperature)
    similarity = similarity.masked_fill(~valid, -1e4)
    probabilities = similarity.softmax(dim=1)
    peak, index = probabilities.max(dim=1)
    side = int(candidates.shape[1] ** 0.5)
    dx = index % side - side // 2
    dy = index // side - side // 2
    near = dx.square() + dy.square() <= max_shift**2
    return near & (peak >= min_peak_probability) & valid.any(dim=1), peak


class DerivedFeatureVerifier:
    """Verify only derived edges; leave their coordinates and direct rows intact.

    The fine maps are paired backbone outputs from the original direction. The
    checkpoint's down_proj/merge_feat layers add the same coarse context used
    by the first fine-matching pass. No new parameters are learned.
    """

    def __init__(
        self,
        pair_matcher,
        states: Dict[str, dict],
        *,
        radius: int = 3,
        max_shift: float = 2.0,
        min_peak_probability: float = 0.12,
        chunk_size: int = 256,
    ):
        if radius < 1 or chunk_size < 1 or max_shift < 0 or not 0 < min_peak_probability < 1:
            raise ValueError("Invalid derived feature verification settings.")
        self.preprocessing = pair_matcher.fine_preprocess
        self.temperature = float(pair_matcher.fine_matching.temperature)
        if self.temperature <= 0:
            raise ValueError("The trained fine-matching temperature must be positive.")
        self.states = states
        self.radius = radius
        self.max_shift = max_shift
        self.min_peak_probability = min_peak_probability
        self.chunk_size = chunk_size

    @torch.no_grad()
    def _verify_pair(self, pair, rows):
        if len(rows) == 0:
            return np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0)
        state = self.states[pair]
        data = state["data"]
        fine0, fine1 = state["feat_f0"], state["feat_f1"]
        coarse0, coarse1 = state["feat_c0"], state["feat_c1"]
        hw0, hw1 = data["hw0_i"], data["hw1_i"]
        offsets = torch.arange(-self.radius, self.radius + 1, device=fine0.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        shifts = torch.stack((ox.flatten(), oy.flatten()), dim=-1)
        source_scale = torch.tensor(
            [hw0[1] / fine0.shape[-1], hw0[0] / fine0.shape[-2]], device=fine0.device
        )
        target_scale = torch.tensor(
            [hw1[1] / fine1.shape[-1], hw1[0] / fine1.shape[-2]], device=fine0.device
        )
        decisions, forward_peaks, reverse_peaks = [], [], []
        for start in range(0, len(rows), self.chunk_size):
            chunk = torch.as_tensor(
                rows[start : start + self.chunk_size, :4], device=fine0.device, dtype=torch.float32
            )
            src, dst = chunk[:, :2], chunk[:, 2:4]
            target_positions = dst[:, None, :] + shifts[None] * target_scale
            source_positions = src[:, None, :] + shifts[None] * source_scale
            valid_target = (
                (target_positions[..., 0] >= 0)
                & (target_positions[..., 0] <= hw1[1] - 1)
                & (target_positions[..., 1] >= 0)
                & (target_positions[..., 1] <= hw1[0] - 1)
            )
            valid_source = (
                (source_positions[..., 0] >= 0)
                & (source_positions[..., 0] <= hw0[1] - 1)
                & (source_positions[..., 1] >= 0)
                & (source_positions[..., 1] <= hw0[0] - 1)
            )
            source_context, target_context = _pair_context(
                src, dst, coarse0, coarse1, data, self.preprocessing
            )
            source_feature = _merge_context(
                _sample(fine0, src, hw0), source_context, self.preprocessing
            )
            target_feature = _merge_context(
                _sample(fine1, dst, hw1), target_context, self.preprocessing
            )
            target_candidates = _merge_context(
                _sample(fine1, target_positions, hw1), target_context, self.preprocessing
            )
            source_candidates = _merge_context(
                _sample(fine0, source_positions, hw0), source_context, self.preprocessing
            )
            forward, peak_f = _direction_test(
                source_feature,
                target_candidates,
                valid_target,
                self.temperature,
                self.min_peak_probability,
                self.max_shift,
            )
            reverse, peak_r = _direction_test(
                target_feature,
                source_candidates,
                valid_source,
                self.temperature,
                self.min_peak_probability,
                self.max_shift,
            )
            decisions.append((forward & reverse).cpu().numpy())
            forward_peaks.append(peak_f.cpu().numpy())
            reverse_peaks.append(peak_r.cpu().numpy())
        return (
            np.concatenate(decisions),
            np.concatenate(forward_peaks),
            np.concatenate(reverse_peaks),
        )

    def __call__(self, result):
        masks, diagnostics = {}, {}
        for pair in ("rs", "si", "ir"):
            rows = result["enhanced"][pair]
            derived = result["derived_mask"][pair]
            derived_indices = np.flatnonzero(derived)
            passed, forward_peak, reverse_peak = self._verify_pair(pair, rows[derived])
            keep = np.ones(len(rows), dtype=bool)
            keep[derived_indices] = passed
            masks[pair] = keep
            diagnostics[pair] = {
                "candidates": int(len(derived_indices)),
                "accepted": int(passed.sum()),
                "rejected": int((~passed).sum()),
                "mean_forward_peak": float(forward_peak.mean()) if len(forward_peak) else None,
                "mean_reverse_peak": float(reverse_peak.mean()) if len(reverse_peak) else None,
            }
        result["feature_verified_mask"] = masks
        result["derived_feature_verification"] = diagnostics
        return result
