"""TCM-Fine: local fine matching, subpixel regression and tri-modal refinement.

The learned pairwise fine matcher is shared across modalities 1->2, 2->3 and 3->1.
Its outputs feed cycle correction, completion and final geometry verification.
The array-only entry point remains available for inspecting the cycle stage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from cycle.refinement import refine_tri_cycle_matches


class TCMFinePreprocess(nn.Module):
    """Build local fine windows around the shared pairwise coarse matches."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config["fine_concat_coarse_feat"]
        self.W = self.config["fine_window_size"]

        # d_model_c = self.config['coarse']['d_model']
        # d_model_f = self.config['fine']['d_model']
        d_model_c = 192
        d_model_f = 96
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data["hw0_f"][0] // data["hw0_c"][0]

        data.update({"W": W})
        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f0_unfold = rearrange(feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f1_unfold = rearrange(feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]

        # option: use coarse-level pair_matcher feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(
                torch.cat(
                    [feat_c0[data["b_ids"], data["i_ids"]], feat_c1[data["b_ids"], data["j_ids"]]],
                    0,
                )
            )  # [2n, c]
            feat_cf_win = self.merge_feat(
                torch.cat(
                    [
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        repeat(feat_c_win, "n c -> n ww c", ww=W**2),  # [2n, ww, cf]
                    ],
                    -1,
                )
            )
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold


class TCMFineMatching(nn.Module):
    """Match local fine windows and regress subpixel offsets for each pair."""

    def __init__(self, config):
        super().__init__()
        self.temperature = config["fine"]["dsmax_temperature"]
        self.W_f = config["fine_window_size"]
        self.fine_thr = config["fine"]["thr"]
        # dim_f = config['resnetfpn']['block_dims'][0]
        dim_f = 96
        self.subpixel_conv_block = nn.Sequential(
            nn.Conv2d(2 * dim_f, 2 * dim_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim_f, 2 * dim_f, kernel_size=3, stride=3, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim_f, 4, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, feat_f0_unfold, feat_f1_unfold, feat_f0_whole, feat_f1_whole, data):
        """
        Args:
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            feat_f0_whole (torch.Tensor): [N, C, H0_f, W0_f]
            feat_f1_whole (torch.Tensor): [N, C, H1_f, W1_f]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0_unfold.shape

        # corner case: if no coarse matches found
        if M == 0:
            assert not self.training, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update(
                {
                    "mkpts0_f": data["mkpts0_c"],
                    "mkpts1_f": data["mkpts1_c"],
                    "mconf_f": torch.zeros(0, device=feat_f0_unfold.device),
                    "fine_anchor_ids": data["coarse_anchor_ids"],
                }
            )
            return

        # normalize
        feat_f0, feat_f1 = map(
            lambda feat: feat / feat.shape[-1] ** 0.5, [feat_f0_unfold, feat_f1_unfold]
        )
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0, feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update({"conf_matrix_fine": conf_matrix_fine})

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(
            **self.get_fine_match(
                conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, feat_f0_whole, feat_f1_whole, data
            )
        )

    def get_fine_match(
        self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, feat_f0_whole, feat_f1_whole, data
    ):
        """
        Args:
            conf_matrix_fine (torch.Tensor): [M, WW, WW]
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            feat_f0_whole (torch.Tensor): [N, C, H0_f, W0_f]
            feat_f1_whole (torch.Tensor): [N, C, H1_f, W1_f]
            data (dict)
        Update:
            data (dict):{
                'm_bids' (torch.Tensor): [M]
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        W_f = self.W_f

        # Selection is discrete, but the later feature gathering and subpixel
        # convolution must remain in the graph for the reprojection loss.
        with torch.no_grad():
            mask = conf_matrix_fine > self.fine_thr
            fallback = not mask.any()
            if fallback:
                mask[0, 0, 0] = True
            mask &= conf_matrix_fine == conf_matrix_fine.amax(dim=(1, 2), keepdim=True)
            # Preserve the fallback even when its confidence is below the
            # threshold; never overwrite conf_matrix_fine in-place.
            if not mask.any():
                mask[0, 0, 0] = True
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix_fine[b_ids, i_ids, j_ids]
        if fallback:
            # Match the historical no-candidate output confidence without
            # modifying the probability tensor needed by TCMFine loss.
            mconf = torch.ones_like(mconf)

        # 4. update with matches in original image resolution

        # indices from coarse matches
        b_ids_c, i_ids_c, j_ids_c = data["b_ids"], data["i_ids"], data["j_ids"]

        # scale (coarse level / fine-level)
        scale_f_c = data["hw0_f"][0] // data["hw0_c"][0]

        # coarse level matches scaled to fine-level (1/2)
        mkpts0_c_scaled_to_f = (
            torch.stack(
                [
                    i_ids_c % data["hw0_c"][1],
                    torch.div(i_ids_c, data["hw0_c"][1], rounding_mode="trunc"),
                ],
                dim=1,
            )
            * scale_f_c
        )

        mkpts1_c_scaled_to_f = (
            torch.stack(
                [
                    j_ids_c % data["hw1_c"][1],
                    torch.div(j_ids_c, data["hw1_c"][1], rounding_mode="trunc"),
                ],
                dim=1,
            )
            * scale_f_c
        )

        # updated b_ids after second thresholding
        updated_b_ids = b_ids_c[b_ids]

        # scales (image res / fine level)
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        scale0 = scale * data["scale0"][updated_b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][updated_b_ids] if "scale1" in data else scale

        # fine-level discrete matches on window coordiantes
        mkpts0_f_window = torch.stack(
            [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode="trunc")], dim=1
        )

        mkpts1_f_window = torch.stack(
            [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode="trunc")], dim=1
        )

        # unfold(crop) local windows
        mkpts0_f_unscaled = mkpts0_c_scaled_to_f[b_ids] + mkpts0_f_window - (W_f // 2)
        mkpts1_f_unscaled = mkpts1_c_scaled_to_f[b_ids] + mkpts1_f_window - (W_f // 2)

        i_ids_sub = mkpts0_f_unscaled[:, 1] * data["hw0_f"][1] + mkpts0_f_unscaled[:, 0]
        j_ids_sub = mkpts1_f_unscaled[:, 1] * data["hw1_f"][1] + mkpts1_f_unscaled[:, 0]

        feat_sub0_unfold = F.unfold(feat_f0_whole, kernel_size=3, stride=1, padding=1)
        feat_sub0_unfold = rearrange(feat_sub0_unfold, "n (c ww) l -> n l ww c", ww=9)
        feat_sub1_unfold = F.unfold(feat_f1_whole, kernel_size=3, stride=1, padding=1)
        feat_sub1_unfold = rearrange(feat_sub1_unfold, "n (c ww) l -> n l ww c", ww=9)

        feat_sub0_unfold = feat_sub0_unfold[updated_b_ids, i_ids_sub]
        feat_sub0_unfold = rearrange(feat_sub0_unfold, "m (h w) c -> m c h w", h=3, w=3)
        feat_sub1_unfold = feat_sub1_unfold[updated_b_ids, j_ids_sub]
        feat_sub1_unfold = rearrange(feat_sub1_unfold, "m (h w) c -> m c h w", h=3, w=3)

        # sub-pixel refinement
        sub_ref = (
            self.subpixel_conv_block(torch.cat([feat_sub0_unfold, feat_sub1_unfold], dim=1))
            .squeeze(3)
            .squeeze(2)
        )
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=-1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f_train = (mkpts0_f_unscaled + sub_ref0) * scale0
        mkpts1_f_train = (mkpts1_f_unscaled + sub_ref1) * scale1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches are the current prediction (for visualization)
        fine_matches = {
            "m_bids": b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            "mkpts0_f": mkpts0_f[mconf != 0],
            "mkpts1_f": mkpts1_f[mconf != 0],
            "mconf_f": mconf[mconf != 0],
            "fine_anchor_ids": data["coarse_anchor_ids"][b_ids[mconf != 0]],
            "mkpts0_f_train": mkpts0_f_train[mconf != 0],
            "mkpts1_f_train": mkpts1_f_train[mconf != 0],
        }

        return fine_matches


def _fine_array(data: Dict[str, Any]) -> np.ndarray:
    """Convert one pairwise fine output to image-resolution correspondence rows."""
    n = len(data["mkpts0_f"])
    if n == 0 or len(data.get("mconf_f", [])) != n:
        return np.empty((0, 5), dtype=np.float64)
    to_np = lambda tensor: tensor.detach().float().cpu().numpy()  # noqa: E731
    return np.column_stack(
        (to_np(data["mkpts0_f"]), to_np(data["mkpts1_f"]), to_np(data["mconf_f"]))
    ).astype(np.float64, copy=False)


# cycle-record index key -> direction label.
_CYCLE_INDEX_KEY = {"rs": "rs_index", "si": "si_index", "ir": "ir_index"}


def _cycle_member_mask(result: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-direction bool mask marking rows that participate in >=1 valid cycle.

    Working-cycle indices reference surviving direct rows at the front of the
    enhanced arrays. The public ``cycles`` retain original baseline indices.
    """
    enhanced = result.get("prefilter", result["enhanced"])
    member = {pair: np.zeros(len(enhanced[pair]), dtype=bool) for pair in ("rs", "si", "ir")}
    for cyc in result.get("working_cycles", result["cycles"]):
        for pair, key in _CYCLE_INDEX_KEY.items():
            member[pair][cyc[key]] = True
    return member


def _fit_geometry(matches, ransac_thresh, cv2):
    """Estimate one homography for a direction, never one per small track."""
    n = len(matches)
    rejected = np.zeros(n, dtype=bool)
    info = {"status": "insufficient_matches", "homography": None}
    if n < 4:
        return rejected, info
    src = matches[:, :2].astype(np.float64)
    dst = matches[:, 2:4].astype(np.float64)
    for points in (src, dst):
        if len(np.unique(points, axis=0)) < 4 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
            info["status"] = "degenerate_points"
            return rejected, info
    try:
        H, mask = cv2.findHomography(
            src,
            dst,
            cv2.USAC_MAGSAC,
            ransac_thresh,
        )
    except cv2.error:
        H = mask = None
    if H is None or mask is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        info["status"] = "estimation_failed"
        return rejected, info
    projection = np.column_stack((src, np.ones(n))) @ H.T
    valid = np.isfinite(projection).all(axis=1) & (np.abs(projection[:, 2]) > 1e-12)
    errors = np.full(n, np.inf)
    errors[valid] = np.linalg.norm(
        projection[valid, :2] / projection[valid, 2:] - dst[valid], axis=1
    )
    inlier = mask.ravel().astype(bool) & (errors <= ransac_thresh)
    if inlier.sum() < 4:
        info["status"] = "insufficient_inliers"
        return rejected, info
    info.update(status="ok", homography=H)
    return inlier, info


def _geometric_filter(
    result: Dict[str, Any], ransac_thresh: float, *, protect_cycle_members: bool = False
) -> Dict[str, Any]:
    """Verify all output rows using per-direction MAGSAC.

    Default: no cycle exemption. Failed/degenerate estimation yields no verified
    rows, with candidates retained in ``prefilter`` and a status in ``geometry``.
    ``cycles`` retains original baseline indices, while ``working_cycles``
    addresses prefilter rows after evidence pruning. ``output_cycles`` contains
    complete surviving triangles with final output indices. Legacy protection
    is an explicit internal comparison option, not the normal inference policy.
    """
    if not np.isfinite(ransac_thresh) or ransac_thresh <= 0:
        raise ValueError("filter_thresh must be a finite positive number.")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise ImportError("Geometric filtering requires OpenCV (cv2).") from exc

    if "prefilter" not in result:
        result["prefilter"] = {pair: rows.copy() for pair, rows in result["enhanced"].items()}
        result["prefilter_derived_mask"] = {
            pair: mask.copy() for pair, mask in result["derived_mask"].items()
        }
        result["prefilter_edge_support"] = {
            pair: result.get("edge_support", {})
            .get(pair, np.ones(len(rows), dtype=np.uint8))
            .copy()
            for pair, rows in result["enhanced"].items()
        }
    enhanced = result["enhanced"]
    derived = result["derived_mask"]
    result.setdefault("edge_support", {})
    member = _cycle_member_mask(result)
    result["geometry"] = {}
    result["filter_masks"] = {}
    result["prefilter_indices"] = {}
    row_maps = {}

    for pair in ("rs", "si", "ir"):
        matches = result["prefilter"][pair]
        n = len(matches)
        keep, info = _fit_geometry(matches, ransac_thresh, cv2)
        cycle_outliers = member[pair] & ~keep
        info["cycle_outliers"] = int(cycle_outliers.sum())
        info["protected_cycle_outliers"] = int(cycle_outliers.sum()) if protect_cycle_members else 0
        if protect_cycle_members:
            keep |= member[pair]
        enhanced[pair] = matches[keep]
        derived[pair] = result["prefilter_derived_mask"][pair][keep]
        result["edge_support"][pair] = result["prefilter_edge_support"][pair][keep]
        result["geometry"][pair] = info
        result["filter_masks"][pair] = keep
        result["prefilter_indices"][pair] = np.flatnonzero(keep)
        row_maps[pair] = np.full(n, -1, dtype=np.int64)
        row_maps[pair][keep] = np.arange(keep.sum())
        stats = result["stats"]
        stats[f"prefilter_{pair}"] = n
        stats[f"prefilter_derived_{pair}"] = int(result["prefilter_derived_mask"][pair].sum())
        stats[f"filtered_{pair}"] = stats[f"enhanced_{pair}"] = int(keep.sum())
        stats[f"removed_{pair}"] = int((~keep).sum())
        stats[f"derived_{pair}"] = int(derived[pair].sum())
        for level in (1, 2, 3):
            stats[f"output_support_{level}_{pair}"] = int(
                (result["edge_support"][pair] == level).sum()
            )

    output_cycles = []
    for cycle in result.get("working_cycles", result["cycles"]):
        if any(
            row_maps[pair][cycle[key]] < 0 or result["prefilter_derived_mask"][pair][cycle[key]]
            for pair, key in _CYCLE_INDEX_KEY.items()
        ):
            continue
        output_cycle = dict(cycle)
        for pair, key in _CYCLE_INDEX_KEY.items():
            output_cycle[f"baseline_{key}"] = cycle[key]
            output_cycle[key] = int(row_maps[pair][cycle[key]])
        output_cycles.append(output_cycle)
    result["output_cycles"] = output_cycles
    result["stats"]["n_output_cycles"] = len(output_cycles)
    return result


class TCMFine:
    """Run shared pairwise fine matching, then tri-modal cycle refinement.

    The paired matcher keeps the learned fine submodules under their existing
    state-dict keys so previously trained checkpoints remain loadable.
    """

    def __init__(
        self,
        epsilon: float = 5.0,
        tau: float = 8.0,
        sigma: float = 5.0,
        augment_radius: float = 3.0,
        boost: float = 1.0,
        anchor: Optional[str] = "confidence",
        anchor_strength: float = 1.0,
        do_correction: bool = True,
        do_augmentation: bool = True,
        do_weighting: bool = False,
        do_filter: bool = True,
        filter_thresh: float = 3.0,
        # --- cycle discovery: top-K + three-residual Q + one-to-one ---
        sigma_s: float = 4.0,
        sigma_i: float = 4.0,
        sigma_r: float = 6.0,
        top_k: int = 5,
        min_cycle_score: float = 0.15,
        one_to_one: bool = True,
        # --- adaptive correction ---
        lam_si: float = 1.0,
        max_correction: float = 4.0,
        # --- trusted augmentation ---
        min_edge_conf: float = 0.0,
        link_sigma: Optional[float] = None,
        check_target: bool = False,
        local_verify: bool = False,
        local_k: int = 12,
        local_thresh: float = 4.0,
        replace_direct: bool = False,
        replace_margin: float = 0.0,
        # --- cycle-graph options ---
        use_node_clustering: bool = True,
        cluster_radius_r: float = 2.5,
        cluster_radius_s: float = 2.5,
        cluster_radius_i: float = 2.5,
        cluster_strategy: str = "highest_conf",
        use_triangle_scoring: bool = True,
        lambda_s: float = 0.4,
        lambda_i: float = 0.4,
        lambda_r: float = 1.0,
        lambda_d: float = 2.0,
        score_min: float = -10.0,
        score_max: float = 0.0,
        use_topk_augmentation: bool = True,
        topk_k: int = 3,
        topk_lambda_geo: float = 0.1,
        route_by_support: bool = True,
        do_evidence_pruning: bool = True,
        deleted_edge_radius: float = 5.0,
    ) -> None:
        """Build the fine-stage enhancer.

        Parameters
        ----------
        epsilon, tau, sigma, augment_radius, boost:
            Cycle thresholds in image pixels (fine resolution). Defaults match
            :func:`refine_tri_cycle_matches`, which were tuned for this stage.
        anchor, anchor_strength, lam_si, max_correction:
            Correction policy. Default ``anchor='confidence'`` fuses each vertex
            from its two redundant estimates weighted by *link confidence* (no
            fixed strong-branch prior); the cycle quality ``Q`` scales only how
            far a point may move and every move is capped at ``max_correction``
            px. ``lam_si=1.0`` applies no SAR-IR prior (set >1 to trust it more,
            <1 less). ``anchor='si'`` restores the old hard-anchored behaviour;
            ``anchor=None`` averages symmetrically.
        sigma_s, sigma_i, sigma_r, top_k, min_cycle_score, one_to_one:
            Cycle-discovery controls. ``top_k`` candidate links are explored at
            each hop (vs the old single nearest neighbour); the three residuals
            (SAR-link, IR-link, RGB-closure) are scored with their own Gaussian
            scales into a quality ``Q``; candidates with ``Q < min_cycle_score``
            are rejected; ``one_to_one`` assigns greedily best-``Q``-first so each
            match joins at most one cycle.
        do_correction, do_augmentation, do_weighting:
            Toggle each cycle mechanism independently for ablation.
        do_filter, filter_thresh:
            Geometry verification is enabled by default for every output row,
            including triangle members. Set ``do_filter=False`` only for ablation.
            Failed estimation returns no verified rows; candidates and diagnostic
            statuses remain accessible. The threshold is in image pixels.
        route_by_support:
            Classify original rows before correction: isolated pairs bypass
            refinement, linked rows supply completion, accepted triangles supply
            correction. Derived edges do not count as new triangle evidence.
            With cached feature maps, two-hop completion checks each of the
            ``topk_k`` nearest paths within ``epsilon`` (5 px by default) in
            both directions before score-based source selection. The final B
            check still runs on selected derived edges.
        do_evidence_pruning:
            Keep accepted cycles. Delete an exterior when a same-direction
            cycle edge shares one endpoint within 1 px and diverges at the
            other by more than 4 px. Competing actual two-edge paths use the
            same conflict rule; choose the best bidirectionally verified path
            and remove losing branches while preserving shared winning edges.
            Isolated direct edges remain available.
        min_edge_conf, link_sigma, check_target, local_verify, local_k,
        local_thresh, replace_direct, replace_margin:
            Trusted-augmentation gates. By default augmentation only fills source
            gaps and never touches a direct match. ``min_edge_conf`` requires
            both chained edges to be confident; ``link_sigma`` decays a derived
            confidence by link mismatch; ``check_target`` also rejects target
            collisions; ``local_verify`` checks each candidate against the local
            displacement field of its ``local_k`` neighbours within
            ``local_thresh`` px. ``replace_direct`` (off, preserving the
            "never overwrite a direct match" policy) lets a clearly stronger
            candidate replace a weaker direct match by ``replace_margin``.
        """
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.sigma = sigma
        self.augment_radius = augment_radius
        self.boost = boost
        self.anchor = anchor
        self.anchor_strength = anchor_strength
        self.do_correction = do_correction
        self.do_augmentation = do_augmentation
        self.do_weighting = do_weighting
        self.do_filter = do_filter
        self.filter_thresh = filter_thresh
        self.sigma_s = sigma_s
        self.sigma_i = sigma_i
        self.sigma_r = sigma_r
        self.top_k = top_k
        self.min_cycle_score = min_cycle_score
        self.one_to_one = one_to_one
        self.lam_si = lam_si
        self.max_correction = max_correction
        self.min_edge_conf = min_edge_conf
        self.link_sigma = link_sigma
        self.check_target = check_target
        self.local_verify = local_verify
        self.local_k = local_k
        self.local_thresh = local_thresh
        self.replace_direct = replace_direct
        self.replace_margin = replace_margin
        # Advanced cycle-refinement configuration.
        self.use_node_clustering = use_node_clustering
        self.cluster_radius_r = cluster_radius_r
        self.cluster_radius_s = cluster_radius_s
        self.cluster_radius_i = cluster_radius_i
        self.cluster_strategy = cluster_strategy
        self.use_triangle_scoring = use_triangle_scoring
        self.lambda_s = lambda_s
        self.lambda_i = lambda_i
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d
        self.score_min = score_min
        self.score_max = score_max
        self.use_topk_augmentation = use_topk_augmentation
        self.topk_k = topk_k
        self.topk_lambda_geo = topk_lambda_geo
        self.route_by_support = route_by_support
        self.do_evidence_pruning = do_evidence_pruning
        self.deleted_edge_radius = deleted_edge_radius

    @torch.no_grad()
    def from_coarse_states(
        self,
        pair_matcher,
        states: Dict[str, Dict[str, Any]],
        *,
        verify_derived: bool = True,
        refine_cycles: bool = True,
        verifier_options: Optional[Dict[str, Any]] = None,
        cycle_refiner_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Complete all three coarse branches with fine matching and cycles."""
        from models.cycle_feature_refiner import CycleFeatureRefiner
        from models.derived_feature_verifier import DerivedFeatureVerifier

        matches = {
            pair: _fine_array(pair_matcher.forward_fine(states[pair]))
            for pair in ("rs", "si", "ir")
        }
        verifier = (
            DerivedFeatureVerifier(pair_matcher, states, **(verifier_options or {}))
            if verify_derived or (
                self.do_evidence_pruning
                and hasattr(pair_matcher, "fine_preprocess")
                and hasattr(pair_matcher, "fine_matching")
            )
            else None
        )
        refiner = (
            CycleFeatureRefiner(pair_matcher, states, **(cycle_refiner_options or {}))
            if refine_cycles
            else None
        )
        return self(
            matches["rs"], matches["si"], matches["ir"],
            derived_verifier=verifier if verify_derived else None,
            cycle_refiner=refiner,
            edge_verifier=(
                verifier._verify_pair if self.do_evidence_pruning and verifier is not None else None
            ),
        )

    def __call__(
        self, rs, si, ir, *, derived_verifier=None, cycle_refiner=None, edge_verifier=None
    ):
        """Refine three precomputed [N, 5] match arrays in image coordinates."""
        fine_arrays = {"rs": rs, "si": si, "ir": ir}

        # 2. Cycle correction + augmentation on the sub-pixel fine matches.
        result = refine_tri_cycle_matches(
            fine_arrays["rs"],
            fine_arrays["si"],
            fine_arrays["ir"],
            epsilon=self.epsilon,
            tau=self.tau,
            sigma=self.sigma,
            boost=self.boost,
            augment_radius=self.augment_radius,
            do_weighting=self.do_weighting,
            do_correction=self.do_correction,
            do_augmentation=self.do_augmentation,
            use_node_clustering=self.use_node_clustering,
            cluster_radius_r=self.cluster_radius_r,
            cluster_radius_s=self.cluster_radius_s,
            cluster_radius_i=self.cluster_radius_i,
            cluster_strategy=self.cluster_strategy,
            use_triangle_scoring=self.use_triangle_scoring,
            lambda_s=self.lambda_s,
            lambda_i=self.lambda_i,
            lambda_r=self.lambda_r,
            lambda_d=self.lambda_d,
            score_min=self.score_min,
            score_max=self.score_max,
            use_topk_augmentation=(self.use_topk_augmentation),
            topk_k=self.topk_k,
            topk_lambda_geo=self.topk_lambda_geo,
            anchor=self.anchor,
            anchor_strength=self.anchor_strength,
            sigma_s=self.sigma_s,
            sigma_i=self.sigma_i,
            sigma_r=self.sigma_r,
            top_k=self.top_k,
            min_cycle_score=self.min_cycle_score,
            one_to_one=self.one_to_one,
            lam_si=self.lam_si,
            max_correction=self.max_correction,
            min_edge_conf=self.min_edge_conf,
            link_sigma=self.link_sigma,
            check_target=self.check_target,
            local_verify=self.local_verify,
            local_k=self.local_k,
            local_thresh=self.local_thresh,
            replace_direct=self.replace_direct,
            replace_margin=self.replace_margin,
            route_by_support=self.route_by_support,
            cycle_refiner=cycle_refiner,
            edge_verifier=edge_verifier,
            augmentation_verifier=getattr(derived_verifier, "_verify_pair", None),
            do_evidence_pruning=self.do_evidence_pruning,
            deleted_edge_radius=self.deleted_edge_radius,
        )

        # 3. Built-in geometry verification for all three support groups.
        if self.do_filter:
            result = _geometric_filter(result, self.filter_thresh)

        # A feature test is most useful for borderline geometric inliers.
        # Preserve completions deep inside the inlier band: their frozen feature
        # maps were never trained on these synthesized, off-anchor positions.
        # Only derived rows may be removed; direct and cycle indices stay valid.
        if derived_verifier is not None:
            result = derived_verifier(result)
            result["derived_feature_candidates"] = {}
            result["derived_feature_decisions"] = {}
            for pair in ("rs", "si", "ir"):
                rows = result["enhanced"][pair]
                derived = result["derived_mask"][pair]
                feature_pass = np.asarray(result["feature_verified_mask"][pair], dtype=bool)
                if feature_pass.shape != (len(rows),) or not feature_pass[~derived].all():
                    raise ValueError("Feature verification must only reject derived rows.")
                keep = feature_pass.copy()
                if self.do_filter and result["geometry"][pair]["status"] == "ok":
                    H = result["geometry"][pair]["homography"]
                    source_h = np.column_stack((rows[:, :2], np.ones(len(rows))))
                    projected_h = source_h @ H.T
                    valid = np.abs(projected_h[:, 2]) > 1e-12
                    residual = np.full(len(rows), np.inf)
                    residual[valid] = np.linalg.norm(
                        projected_h[valid, :2] / projected_h[valid, 2:3] - rows[valid, 2:4],
                        axis=1,
                    )
                    keep |= residual <= self.filter_thresh * (11.0 / 12.0)
                result["derived_feature_candidates"][pair] = rows[derived].copy()
                result["derived_feature_decisions"][pair] = keep[derived].copy()
                diagnostics = result["derived_feature_verification"][pair]
                diagnostics["feature_passed"] = diagnostics["accepted"]
                diagnostics["accepted"] = int(keep[derived].sum())
                diagnostics["rejected"] = int((~keep[derived]).sum())
                diagnostics["geometric_band_preserved"] = int(
                    (keep[derived] & ~feature_pass[derived]).sum()
                )
                if self.do_filter:
                    old_indices = result["prefilter_indices"][pair]
                    result["filter_masks"][pair][old_indices[~keep]] = False
                    result["prefilter_indices"][pair] = old_indices[keep]
                result["enhanced"][pair] = rows[keep]
                result["derived_mask"][pair] = derived[keep]
                result["edge_support"][pair] = result["edge_support"][pair][keep]
                stats = result["stats"]
                stats[f"enhanced_{pair}"] = int(keep.sum())
                stats[f"derived_{pair}"] = int(result["derived_mask"][pair].sum())
                stats[f"feature_candidates_{pair}"] = diagnostics["candidates"]
                stats[f"feature_rejected_{pair}"] = diagnostics["rejected"]
                if self.do_filter:
                    stats[f"filtered_{pair}"] = int(keep.sum())
                    stats[f"removed_{pair}"] = stats[f"prefilter_{pair}"] - int(keep.sum())
                    for level in (1, 2, 3):
                        stats[f"output_support_{level}_{pair}"] = int(
                            (result["edge_support"][pair] == level).sum()
                        )

        return result


# Keep the original names importable while training code uses TCMFine names.
# The owning PairMatcher attributes remain stable for checkpoint compatibility.
FinePreprocess = TCMFinePreprocess
FineMatching = TCMFineMatching

__all__ = [
    "TCMFinePreprocess", "TCMFineMatching", "TCMFine",
    "FinePreprocess", "FineMatching",
]
