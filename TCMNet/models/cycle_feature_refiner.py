"""Inference-only local fine-feature refinement and rollback for accepted cycles."""

from __future__ import annotations

import numpy as np
import torch

from models.derived_feature_verifier import _merge_context, _pair_context, _sample

PAIRS = ("rs", "si", "ir")
INDEX = {"rs": "rs_index", "si": "si_index", "ir": "ir_index"}


def _closure(rows):
    """Distances between the two estimates of each modality 1/2/3 vertex."""
    rs, si, ir = rows["rs"], rows["si"], rows["ir"]
    return np.array(
        [
            np.linalg.norm(rs[2:4] - si[:2]),
            np.linalg.norm(si[2:4] - ir[:2]),
            np.linalg.norm(ir[2:4] - rs[:2]),
        ]
    )


class CycleFeatureRefiner:
    """Check cycle correction against original matches, then refine local peaks.

    The already-trained pairwise fine maps and context projection are frozen.
    Every accepted original triangle is treated as an atomic rollback unit.
    Local target updates are accepted only when feature evidence rises and the
    three-domain closure remains within its original tolerances.
    """

    def __init__(
        self,
        pair_matcher,
        states,
        *,
        radius_px=2,
        min_peak_probability=0.12,
        min_edge_gain=0.5,
        rollback_margin=2.0,
        max_local_move=1.0,
        max_from_original=4.0,
        chunk_size=256,
    ):
        if (
            radius_px < 1
            or chunk_size < 1
            or not 0 < min_peak_probability < 1
            or min_edge_gain < 0
            or rollback_margin < 0
            or max_local_move <= 0
            or max_from_original <= 0
        ):
            raise ValueError("Invalid cycle feature refinement settings.")
        self.preprocessing = pair_matcher.fine_preprocess
        self.temperature = float(pair_matcher.fine_matching.temperature)
        if self.temperature <= 0:
            raise ValueError("The trained fine-matching temperature must be positive.")
        self.states = states
        self.radius_px = int(radius_px)
        self.min_peak_probability = float(min_peak_probability)
        self.min_edge_gain = float(min_edge_gain)
        self.rollback_margin = float(rollback_margin)
        self.max_local_move = float(max_local_move)
        self.max_from_original = float(max_from_original)
        self.chunk_size = int(chunk_size)

    def _logits(self, query, candidates):
        return (query.float()[:, None, :] * candidates.float()).sum(-1) / (
            query.shape[-1] * self.temperature
        )

    @torch.no_grad()
    def _score_direction(self, pair, corrected, original):
        """Score old/current endpoints and propose target-side subpixel moves."""
        n = len(corrected)
        score_old = np.zeros(n)
        score_now = np.zeros(n)
        score_new = np.zeros(n)
        proposed = corrected[:, 2:4].copy()
        usable = np.zeros(n, dtype=bool)
        if n == 0:
            return score_old, score_now, score_new, proposed, usable

        state = self.states[pair]
        data = state["data"]
        fine0, fine1 = state["feat_f0"], state["feat_f1"]
        coarse0, coarse1 = state["feat_c0"], state["feat_c1"]
        hw0, hw1 = data["hw0_i"], data["hw1_i"]
        radius = self.radius_px
        axes = torch.arange(-radius, radius + 1, device=fine0.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(axes, axes, indexing="ij")
        offsets = torch.stack((ox.flatten(), oy.flatten()), dim=-1)
        side = 2 * radius + 1
        center = radius * side + radius
        row_index = None

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            current = torch.as_tensor(
                corrected[start:end, :4], device=fine0.device, dtype=torch.float32
            )
            base = torch.as_tensor(
                original[start:end, :4], device=fine0.device, dtype=torch.float32
            )
            src, dst = current[:, :2], current[:, 2:4]
            src_context, dst_context = _pair_context(
                base[:, :2], base[:, 2:4], coarse0, coarse1, data, self.preprocessing
            )
            q = _merge_context(_sample(fine0, src, hw0), src_context, self.preprocessing)
            q_base = _merge_context(
                _sample(fine0, base[:, :2], hw0), src_context, self.preprocessing
            )
            dst_base = _merge_context(
                _sample(fine1, base[:, 2:4], hw1), dst_context, self.preprocessing
            )
            old = (q_base.float() * dst_base.float()).sum(-1) / (q.shape[-1] * self.temperature)

            positions = dst[:, None, :] + offsets[None]
            valid = (
                (positions[..., 0] >= 0)
                & (positions[..., 0] <= hw1[1] - 1)
                & (positions[..., 1] >= 0)
                & (positions[..., 1] <= hw1[0] - 1)
            )
            descriptors = _merge_context(
                _sample(fine1, positions, hw1), dst_context, self.preprocessing
            )
            logits = self._logits(q, descriptors).masked_fill(~valid, -1e4)
            probabilities = logits.softmax(dim=1)
            peak, index = probabilities.max(dim=1)
            now = logits[:, center]
            batch = len(current)
            if row_index is None or len(row_index) != batch:
                row_index = torch.arange(batch, device=fine0.device)
            move = offsets[index].clone()

            # A parabola through the peak and its immediate neighbours yields
            # a bounded subpixel offset; boundary peaks stay on the sampled grid.
            for dim, stride in ((0, 1), (1, side)):
                coordinate = index % side if dim == 0 else index // side
                interior = (coordinate > 0) & (coordinate < side - 1)
                left = logits[row_index, (index - stride).clamp(0, side * side - 1)]
                right = logits[row_index, (index + stride).clamp(0, side * side - 1)]
                middle = logits[row_index, index]
                denominator = left - 2 * middle + right
                offset = torch.where(
                    interior & (denominator < -1e-3),
                    0.5 * (left - right) / denominator.clamp(max=-1e-3),
                    torch.zeros_like(middle),
                ).clamp(-0.5, 0.5)
                move[:, dim] += offset
            candidate = dst + move
            candidate_desc = _merge_context(
                _sample(fine1, candidate, hw1), dst_context, self.preprocessing
            )
            new = (q.float() * candidate_desc.float()).sum(-1) / (q.shape[-1] * self.temperature)

            reverse_positions = src[:, None, :] + offsets[None]
            valid_reverse = (
                (reverse_positions[..., 0] >= 0)
                & (reverse_positions[..., 0] <= hw0[1] - 1)
                & (reverse_positions[..., 1] >= 0)
                & (reverse_positions[..., 1] <= hw0[0] - 1)
            )
            reverse_desc = _merge_context(
                _sample(fine0, reverse_positions, hw0), src_context, self.preprocessing
            )
            reverse_logits = self._logits(candidate_desc, reverse_desc).masked_fill(
                ~valid_reverse, -1e4
            )
            reverse_peak, reverse_index = reverse_logits.softmax(dim=1).max(dim=1)
            reverse_move = offsets[reverse_index]
            passed = (
                (peak >= self.min_peak_probability)
                & (reverse_peak >= self.min_peak_probability)
                & (reverse_move.square().sum(-1) <= 1.0)
                & (new >= now + self.min_edge_gain)
                & (candidate[:, 0] >= 0)
                & (candidate[:, 0] <= hw1[1] - 1)
                & (candidate[:, 1] >= 0)
                & (candidate[:, 1] <= hw1[0] - 1)
                & valid.any(dim=1)
                & valid_reverse.any(dim=1)
            )
            score_old[start:end] = old.cpu().numpy()
            score_now[start:end] = now.cpu().numpy()
            score_new[start:end] = new.cpu().numpy()
            proposed[start:end] = candidate.cpu().numpy()
            usable[start:end] = passed.cpu().numpy()
        return score_old, score_now, score_new, proposed, usable

    @torch.no_grad()
    def __call__(self, enhanced, baseline, cycles, *, epsilon, tau):
        active = [
            cycle
            for cycle in cycles
            if cycle.get("correction_weight", cycle.get("cycle_weight", 0.0)) > 0
            and any(
                np.max(
                    np.abs(
                        enhanced[pair][cycle[INDEX[pair]], :4]
                        - baseline[pair][cycle[INDEX[pair]], :4]
                    )
                )
                > 1e-6
                for pair in PAIRS
            )
        ]
        diagnostics = {
            "cycles_considered": len(active),
            "rolled_back": 0,
            "locally_refined": 0,
            "edges_refined": 0,
            "unchanged": 0,
        }
        if not active:
            return enhanced, diagnostics

        positions = {
            pair: np.array([cycle[INDEX[pair]] for cycle in active], dtype=np.int64)
            for pair in PAIRS
        }
        original = {pair: baseline[pair][positions[pair]].copy() for pair in PAIRS}
        current = {pair: enhanced[pair][positions[pair]].copy() for pair in PAIRS}
        evidence = {
            pair: self._score_direction(pair, current[pair], original[pair]) for pair in PAIRS
        }

        for k, cycle in enumerate(active):
            original_rows = {pair: original[pair][k].copy() for pair in PAIRS}
            current_rows = {pair: current[pair][k].copy() for pair in PAIRS}
            old_score = sum(evidence[pair][0][k] for pair in PAIRS)
            current_score = sum(evidence[pair][1][k] for pair in PAIRS)
            cycle["feature_score_delta"] = float(old_score - current_score)
            if old_score > current_score + self.rollback_margin:
                for pair in PAIRS:
                    enhanced[pair][positions[pair][k], :4] = original_rows[pair][:4]
                cycle["feature_action"] = "rollback"
                diagnostics["rolled_back"] += 1
                continue

            trial = {pair: row.copy() for pair, row in current_rows.items()}
            initial_closure = _closure(current_rows)
            gains = sorted(
                (
                    (float(evidence[pair][2][k] - evidence[pair][1][k]), pair)
                    for pair in PAIRS
                    if evidence[pair][4][k]
                ),
                reverse=True,
            )
            moved = 0
            for _, pair in gains:
                target = evidence[pair][3][k]
                if np.linalg.norm(target - current_rows[pair][2:4]) > self.max_local_move:
                    continue
                if np.linalg.norm(target - original_rows[pair][2:4]) > self.max_from_original:
                    continue
                candidate = {name: row.copy() for name, row in trial.items()}
                candidate[pair][2:4] = target
                distances = _closure(candidate)
                if (
                    distances[0] >= epsilon
                    or distances[1] >= epsilon
                    or distances[2] >= tau
                    or np.sum(distances**2) > np.sum(initial_closure**2) + 0.5
                ):
                    continue
                trial = candidate
                moved += 1
            if moved:
                for pair in PAIRS:
                    enhanced[pair][positions[pair][k], :4] = trial[pair][:4]
                cycle["feature_action"] = "refined"
                diagnostics["locally_refined"] += 1
                diagnostics["edges_refined"] += moved
            else:
                cycle["feature_action"] = "keep_correction"
                diagnostics["unchanged"] += 1
        return enhanced, diagnostics
