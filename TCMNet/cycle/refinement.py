"""Tri-modal cycle-consistency refinement for RGB/SAR/IR matches.

This module retains pairwise matches unless
optional, independent feature evidence confirms a graph conflict or a weak
edge in a non-closing three-edge path. The triangular relation also supplies:

* **Weighting** - matches that fall inside a consistent cycle get their
  confidence boosted; matches outside any cycle keep their original value.
* **Correction** - each modality coordinate has two redundant estimates inside
  a closed cycle (one from each adjacent link). Averaging them inside that
  modality's own frame reduces the cycle-closure error and denoises the point.
* **Augmentation** - a transitive 2-link path can synthesize a correspondence
  the direct matcher missed. These are appended and flagged ``cycle-derived``.

All distances are computed *inside a single modality's own coordinate frame* and
never across modalities, so the logic is invariant to flips/crops/scales between
modalities: it measures whether going once around the three modalities returns
to the start, which is decoupled from the inter-modal geometry.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "tri_cycle_refinement requires SciPy. Install it with: pip install scipy"
    ) from exc

from .augmentation import transitive_chain_candidates, transitive_chain_topk
from .graph import (
    classify_edge_support,
    cluster_trimodal_matches,
    cycle_weight_from_score,
    triangle_score,
)


def _validate_matches(matches: Any, name: str) -> np.ndarray:
    """Convert one match collection to a finite float64 array of shape (N, 5)."""
    array = np.asarray(matches, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(
            f"{name} must have shape (N, 5), but received {array.shape}. "
            "Each row must contain four coordinates and one confidence."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(array[:, 4] < 0):
        raise ValueError(f"{name} contains a negative confidence value.")
    return array


def _validate_parameters(
    epsilon: float,
    tau: float,
    sigma: float,
    boost: float,
    augment_radius: float,
) -> None:
    for value, vname in (
        (epsilon, "epsilon"),
        (tau, "tau"),
        (sigma, "sigma"),
        (augment_radius, "augment_radius"),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{vname} must be a finite positive number.")
    if not np.isscalar(boost) or not np.isfinite(boost) or boost < 0:
        raise ValueError("boost must be a finite non-negative number.")


def _geometric_mean3(a: float, b: float, c: float) -> float:
    """Numerically stable three-value geometric mean (0 if any factor is 0)."""
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return 0.0
    return float(np.exp((np.log(a) + np.log(b) + np.log(c)) / 3.0))


def _query_neighbors(
    tree: cKDTree, point: np.ndarray, k: int, radius: float
) -> List[Tuple[float, int]]:
    """Return up to ``k`` neighbours of ``point`` strictly within ``radius``.

    A KD-tree query with ``distance_upper_bound`` pads missing neighbours with an
    infinite distance and an out-of-range index; those are filtered out so only
    real hits inside the radius come back, sorted nearest-first.
    """
    dist, idx = tree.query(point, k=k, distance_upper_bound=radius)
    dist = np.atleast_1d(dist)
    idx = np.atleast_1d(idx)
    out: List[Tuple[float, int]] = []
    for d, j in zip(dist, idx):
        if np.isfinite(d) and int(j) < tree.n and d < radius:
            out.append((float(d), int(j)))
    return out


def _cycle_weight_from_residuals(
    d_s: float,
    d_i: float,
    d_r: float,
    sigma: float,
    sigma_s: Optional[float],
    sigma_i: Optional[float],
    sigma_r: Optional[float],
) -> float:
    """Map the three loop residuals to a ``[0, 1]`` Gaussian quality weight.

    Backward-compatible: when the per-modality scales ``sigma_s``/``sigma_i`` are
    left as ``None`` the weight collapses to the legacy single-residual form
    ``exp(-d_r^2 / 2 sigma^2)`` (only the RGB closure error matters). When all
    three scales are supplied the full three-residual form is used, so the SAR
    and IR link distances also shape the score instead of being mere gates.
    """
    if sigma_s is None or sigma_i is None:
        return float(np.exp(-(d_r**2) / (2.0 * float(sigma) ** 2)))
    s_r = float(sigma_r) if sigma_r is not None else float(sigma)
    exponent = (
        (d_s**2) / (2.0 * float(sigma_s) ** 2)
        + (d_i**2) / (2.0 * float(sigma_i) ** 2)
        + (d_r**2) / (2.0 * s_r**2)
    )
    return float(np.exp(-exponent))


def _find_cycles(
    rs: np.ndarray,
    si: np.ndarray,
    ir: np.ndarray,
    epsilon: float,
    tau: float,
    sigma: float,
    sigma_s: Optional[float] = None,
    sigma_i: Optional[float] = None,
    sigma_r: Optional[float] = None,
    top_k: int = 1,
    min_cycle_score: float = 0.0,
    one_to_one: bool = False,
) -> List[Dict[str, Any]]:
    """Discover valid RGB -> SAR -> IR -> RGB cycles over the three match sets.

    A cycle links one row of each set: the S_RS target (SAR) must be within
    ``epsilon`` of an S_SI source (SAR), the S_SI target (IR) must be within
    ``epsilon`` of an S_IR source (IR), and the S_IR target (RGB) must be within
    ``tau`` of the originating S_RS source (RGB). All three comparisons happen
    inside one modality's own frame, so the test is invariant to inter-modal
    flips/crops.

    ``top_k`` controls how many candidate links are explored at each hop instead
    of only the single nearest neighbour: a wrong nearest neighbour can no longer
    hide a real cycle behind it. ``min_cycle_score`` rejects candidates whose
    quality ``Q`` (tri-confidence x Gaussian residual weight) is too low. When
    ``one_to_one`` is set, candidates are accepted greedily highest-``Q`` first
    and each ``rs``/``si``/``ir`` row is consumed by at most one cycle, so several
    RS matches can no longer all latch onto the same SI/IR point.

    With the defaults (``top_k=1``, ``min_cycle_score=0``, ``one_to_one=False``,
    per-modality sigmas ``None``) this reproduces the legacy greedy single-NN
    behaviour exactly, which the coarse-stage module relies on.
    """
    if len(rs) == 0 or len(si) == 0 or len(ir) == 0:
        return []

    sar_tree = cKDTree(si[:, 0:2])  # SAR sources of the SAR->IR links
    ir_tree = cKDTree(ir[:, 0:2])  # IR sources of the IR->RGB links
    candidates: List[Dict[str, Any]] = []

    for rs_index, rs_match in enumerate(rs):
        rgb = rs_match[0:2]
        sar = rs_match[2:4]  # SAR estimate from the RGB->SAR link
        conf_rs = float(rs_match[4])

        for sar_distance, si_index in _query_neighbors(sar_tree, sar, top_k, epsilon):
            si_match = si[si_index]
            sar_hat = si_match[0:2]  # SAR estimate from the SAR->IR link
            ir_coord = si_match[2:4]  # IR estimate from the SAR->IR link
            conf_si = float(si_match[4])

            for ir_distance, ir_index in _query_neighbors(ir_tree, ir_coord, top_k, epsilon):
                ir_match = ir[ir_index]
                ir_hat = ir_match[0:2]  # IR estimate from the IR->RGB link
                rgb_hat = ir_match[2:4]  # RGB estimate from the IR->RGB link
                conf_ir = float(ir_match[4])

                cycle_error = float(np.linalg.norm(rgb - rgb_hat))
                if cycle_error >= tau:
                    continue

                tri_confidence = _geometric_mean3(conf_rs, conf_si, conf_ir)
                cycle_weight = _cycle_weight_from_residuals(
                    sar_distance,
                    ir_distance,
                    cycle_error,
                    sigma,
                    sigma_s,
                    sigma_i,
                    sigma_r,
                )
                score = tri_confidence * cycle_weight
                if score < min_cycle_score:
                    continue

                candidates.append(
                    {
                        "rs_index": rs_index,
                        "si_index": si_index,
                        "ir_index": ir_index,
                        # Two redundant estimates per modality, each in its own frame.
                        "rgb": rgb.copy(),  # RGB from RS source
                        "rgb_hat": rgb_hat.copy(),  # RGB from IR target
                        "sar": sar.copy(),  # SAR from RS target
                        "sar_hat": sar_hat.copy(),  # SAR from SI source
                        "ir": ir_coord.copy(),  # IR from SI target
                        "ir_hat": ir_hat.copy(),  # IR from IR source
                        "conf_rs": conf_rs,
                        "conf_si": conf_si,
                        "conf_ir": conf_ir,
                        "sar_link_distance": float(sar_distance),
                        "ir_link_distance": float(ir_distance),
                        "cycle_error": cycle_error,
                        "score": score,
                        "quality": score,
                        "cycle_weight": cycle_weight,
                        "tri_confidence": tri_confidence,
                    }
                )

    if not one_to_one:
        return candidates

    # Greedy one-to-one assignment: best Q first, each row used by one cycle only.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    used = {"rs": set(), "si": set(), "ir": set()}
    accepted: List[Dict[str, Any]] = []
    for cand in candidates:
        if (
            cand["rs_index"] in used["rs"]
            or cand["si_index"] in used["si"]
            or cand["ir_index"] in used["ir"]
        ):
            continue
        used["rs"].add(cand["rs_index"])
        used["si"].add(cand["si_index"])
        used["ir"].add(cand["ir_index"])
        accepted.append(cand)
    return accepted


def _apply_weighting(
    enhanced: Dict[str, np.ndarray],
    cycles: List[Dict[str, Any]],
    boost: float,
) -> Dict[str, np.ndarray]:
    """Scale confidence of rows participating in a cycle by ``1 + boost * w``.

    A row may take part in several cycles; the strongest cycle weight wins. Rows
    outside every cycle keep their original confidence unchanged.
    """
    best_weight = {
        "rs": np.zeros(len(enhanced["rs"])),
        "si": np.zeros(len(enhanced["si"])),
        "ir": np.zeros(len(enhanced["ir"])),
    }
    index_key = {"rs": "rs_index", "si": "si_index", "ir": "ir_index"}
    for cyc in cycles:
        w = cyc["cycle_weight"]
        for pair, key in index_key.items():
            idx = cyc[key]
            if w > best_weight[pair][idx]:
                best_weight[pair][idx] = w

    for pair in ("rs", "si", "ir"):
        if len(enhanced[pair]) == 0:
            continue
        enhanced[pair][:, 4] = enhanced[pair][:, 4] * (1.0 + boost * best_weight[pair])
    return enhanced


def _fuse_pair(a: np.ndarray, conf_a: float, b: np.ndarray, conf_b: float) -> np.ndarray:
    """Confidence-weighted fusion of two estimates of the same point.

    The mixing ratio depends only on the two confidences (never on the cycle
    quality Q): a high-confidence estimate dominates, a low one yields. With
    equal confidences this is the plain midpoint. Falls back to the midpoint if
    both confidences are non-positive.
    """
    w = conf_a + conf_b
    if w <= 0.0:
        return (a + b) / 2.0
    return (conf_a * a + conf_b * b) / w


def _clamp_step(orig: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Limit the correction ``orig -> target`` to at most ``max_step`` pixels.

    A wrong cycle that wants to drag a point far away is capped, so correction
    can only ever nudge a coordinate, never teleport it.
    """
    if max_step is None or max_step <= 0.0:
        return target
    delta = target - orig
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist == 0.0:
        return target
    return orig + delta * (max_step / dist)


def _apply_correction(
    enhanced: Dict[str, np.ndarray],
    cycles: List[Dict[str, Any]],
    anchor: Optional[str] = None,
    anchor_strength: float = 1.0,
    lam_si: float = 1.0,
    max_correction: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Correct cycle coordinates using the redundant per-modality estimates.

    Correction policies:

    * ``anchor=None`` (symmetric): each modality coordinate is replaced by the
      mean of its two redundant estimates. Treats all three branches equally.
    * ``anchor='si'`` (SAR-IR anchored): the SAR-IR branch is the strong,
      structurally-consistent reference and is left untouched. The two weak
      branches (RGB-SAR, IR-RGB) have only their endpoint *shared with the SI
      branch* pulled toward the SI estimate, by ``anchor_strength`` in [0, 1]
      (1.0 = hard snap onto the SI anchor). The RGB vertex has no strong
      reference and is never moved.
    * ``anchor='confidence'`` (adaptive): no branch is assumed strong a priori.
      Each of the three vertices (RGB included) is fused from its two redundant
      estimates weighted by the *link confidences* alone -- the more confident
      estimate wins, the direction is decided per cycle by the data. The cycle
      quality ``Q`` (``score``) does NOT enter the fusion weights; it only scales
      how far the point is allowed to move (``step = anchor_strength * Q``), so
      low-quality cycles barely nudge their points. ``lam_si`` (default 1.0 = no
      prior) optionally up/down-weights the SAR-IR branch's say in the shared
      SAR and IR vertices. Every move is capped at ``max_correction`` pixels.

    Every comparison stays inside one modality's own frame, so correction is
    valid regardless of inter-modal flips/crops/scales. A row touched by several
    cycles is averaged over all of them.
    """
    if anchor not in (None, "si", "ir", "confidence"):
        raise ValueError("anchor must be None, 'si', 'ir', or 'confidence'.")

    accum: Dict[Tuple[str, int, int], List[np.ndarray]] = {}

    def add(pair: str, row: int, col: int, value: np.ndarray) -> None:
        accum.setdefault((pair, row, col), []).append(value)

    if anchor == "confidence":
        base_s = float(anchor_strength)
        lam = float(lam_si)
        for cyc in cycles:
            c_rs, c_si, c_ir = cyc["conf_rs"], cyc["conf_si"], cyc["conf_ir"]
            # Fuse each vertex from its two redundant estimates (confidence only).
            rgb_star = _fuse_pair(cyc["rgb"], c_rs, cyc["rgb_hat"], c_ir)
            sar_star = _fuse_pair(cyc["sar"], c_rs, cyc["sar_hat"], lam * c_si)
            ir_star = _fuse_pair(cyc["ir"], lam * c_si, cyc["ir_hat"], c_ir)
            # Q modulates only HOW FAR we move toward the fused estimate.
            quality = cyc.get("correction_weight", cyc["score"])
            s = float(np.clip(base_s * quality, 0.0, 1.0))
            targets = {
                ("rs", cyc["rs_index"], 0): (cyc["rgb"], rgb_star),
                ("rs", cyc["rs_index"], 2): (cyc["sar"], sar_star),
                ("si", cyc["si_index"], 0): (cyc["sar_hat"], sar_star),
                ("si", cyc["si_index"], 2): (cyc["ir"], ir_star),
                ("ir", cyc["ir_index"], 0): (cyc["ir_hat"], ir_star),
                ("ir", cyc["ir_index"], 2): (cyc["rgb_hat"], rgb_star),
            }
            for (pair, row, col), (orig, fused) in targets.items():
                moved = orig + s * (fused - orig)
                moved = _clamp_step(orig, moved, max_correction)
                add(pair, row, col, moved)
    elif anchor is None:
        for cyc in cycles:
            rgb_star = (cyc["rgb"] + cyc["rgb_hat"]) / 2.0
            sar_star = (cyc["sar"] + cyc["sar_hat"]) / 2.0
            ir_star = (cyc["ir"] + cyc["ir_hat"]) / 2.0
            rs_i, si_i, ir_i = cyc["rs_index"], cyc["si_index"], cyc["ir_index"]
            # RS row: source = RGB (cols 0:2), target = SAR (cols 2:4).
            add("rs", rs_i, 0, rgb_star)
            add("rs", rs_i, 2, sar_star)
            # SI row: source = SAR (cols 0:2), target = IR (cols 2:4).
            add("si", si_i, 0, sar_star)
            add("si", si_i, 2, ir_star)
            # IR row: source = IR (cols 0:2), target = RGB (cols 2:4).
            add("ir", ir_i, 0, ir_star)
            add("ir", ir_i, 2, rgb_star)
    elif anchor == "si":  # SAR-IR anchored: weak branches pulled onto the strong SI branch
        a = float(anchor_strength)
        for cyc in cycles:
            # SI is the anchor: sar_hat (SI source) and ir (SI target) are trusted.
            new_sar = a * cyc["sar_hat"] + (1.0 - a) * cyc["sar"]
            new_ir = a * cyc["ir"] + (1.0 - a) * cyc["ir_hat"]
            # RS weak branch: only its SAR endpoint (cols 2:4) is anchored.
            add("rs", cyc["rs_index"], 2, new_sar)
            # IR weak branch: only its IR endpoint (cols 0:2) is anchored.
            add("ir", cyc["ir_index"], 0, new_ir)
            # SI branch and both RGB vertices are left untouched.
    else:  # anchor == 'ir' -- IR-RGB is strong; only the shared SAR vertex is uncertain
        for cyc in cycles:
            # Average the two independent SAR estimates to reduce uncertainty.
            new_sar = (cyc["sar"] + cyc["sar_hat"]) / 2.0
            # RS weak branch: update its SAR target (cols 2:4).
            add("rs", cyc["rs_index"], 2, new_sar)
            # SI weak branch: update its SAR source (cols 0:2).
            add("si", cyc["si_index"], 0, new_sar)
            # IR branch, IR vertices, and RGB vertices are left untouched.

    for (pair, row, col), values in accum.items():
        enhanced[pair][row, col : col + 2] = np.mean(values, axis=0)
    return enhanced


def _passes_local_geometry(
    cand: np.ndarray,
    existing: np.ndarray,
    k: int,
    thresh: float,
) -> bool:
    """Check a candidate against the *local* displacement field of its neighbours.

    A global homography is a poor model for remote-sensing pairs with local
    relief/parallax. Instead we take the ``k`` existing matches nearest the
    candidate's source point, compute the median of their source->target
    displacement vectors, and accept the candidate only when its own
    displacement agrees within ``thresh`` pixels:

        ``e_local = || (cand_tgt - cand_src) - median(neighbour displacements) ||``

    The displacement is the same source->target relation for the candidate and
    its neighbours, so the comparison is internally consistent even when the two
    modalities differ by a global flip (the field is only assumed *locally*
    smooth, which holds over a small neighbourhood). When fewer than ``k``
    neighbours exist the field is undersampled and we fall back to a global
    MAGSAC homography; if neither can be evaluated we do not block the candidate.
    """
    source = cand[0:2]
    disp = cand[2:4] - cand[0:2]
    n = len(existing)
    if n >= k and k > 0:
        tree = cKDTree(existing[:, 0:2])
        _, idxs = tree.query(source, k=k)
        neigh = existing[np.atleast_1d(idxs)]
        median_disp = np.median(neigh[:, 2:4] - neigh[:, 0:2], axis=0)
        return bool(np.linalg.norm(disp - median_disp) < thresh)

    # Undersampled neighbourhood: fall back to a global homography if possible.
    try:
        import cv2
    except ImportError:
        return True  # cannot judge -> do not block
    if n < 4:
        return True
    H, _ = cv2.findHomography(
        existing[:, 0:2].astype(np.float64),
        existing[:, 2:4].astype(np.float64),
        cv2.USAC_MAGSAC,
        thresh,
    )
    if H is None:
        return True
    src_h = np.array([source[0], source[1], 1.0])
    proj = H @ src_h
    proj = proj[0:2] / proj[2]
    return bool(np.linalg.norm(proj - cand[2:4]) < thresh)


_INDEX_KEY = {"rs": "rs_index", "si": "si_index", "ir": "ir_index"}
_EVIDENCE_PATH_JOIN_RADIUS = 0.5  # px; actual consecutive two-edge joins in step ③
_EVIDENCE_CONFLICT_SHARED_RADIUS = 1.0  # px; same-direction shared endpoint
_EVIDENCE_CYCLE_EXTERIOR_GAP = 4.0  # px; accepted-cycle exterior only
_EVIDENCE_PATH_CONFLICT_GAP = 4.0  # px; competing open two-edge paths


def _score_cycles(
    cycles: List[Dict[str, Any]],
    lambda_s: float,
    lambda_i: float,
    lambda_r: float,
    lambda_d: float,
    score_min: float,
    score_max: float,
) -> None:
    """Attach the advanced joint triangle score and confidence weight."""
    for cyc in cycles:
        d_s = float(np.linalg.norm(cyc["sar"] - cyc["sar_hat"]))
        d_i = float(np.linalg.norm(cyc["ir"] - cyc["ir_hat"]))
        d_r = float(np.linalg.norm(cyc["rgb"] - cyc["rgb_hat"]))
        n_derived = (
            int(cyc.get("is_derived_rs", False))
            + int(cyc.get("is_derived_si", False))
            + int(cyc.get("is_derived_ir", False))
        )
        score = triangle_score(
            conf_rs=cyc["conf_rs"],
            conf_si=cyc["conf_si"],
            conf_ir=cyc["conf_ir"],
            d_s=d_s,
            d_i=d_i,
            d_r=d_r,
            lambda_s=lambda_s,
            lambda_i=lambda_i,
            lambda_r=lambda_r,
            lambda_d=lambda_d,
            n_derived=n_derived,
        )
        cyc["triangle_score"] = score
        cyc["cycle_weight"] = cycle_weight_from_score(score, score_min, score_max)
        cyc["correction_weight"] = cyc["cycle_weight"]
        cyc["score"] = score


def _row_to_node_maps(
    node_info: Dict[str, Any],
) -> Dict[str, Dict[int, Tuple[int, int]]]:
    """Invert clustered edge records into matcher-row to physical-node maps."""
    maps: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for pair in ("rs", "si", "ir"):
        row_map: Dict[int, Tuple[int, int]] = {}
        for node_src, node_tgt, row_idx in node_info["edge_nodes"][pair]:
            row_map[int(row_idx)] = (int(node_src), int(node_tgt))
        maps[pair] = row_map
    return maps


def _node_aware_one_to_one(
    cycles: List[Dict[str, Any]],
    node_info: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Greedily assign cycles with row- or physical-node-level conflicts."""
    ordered = sorted(cycles, key=lambda cycle: cycle["score"], reverse=True)
    if node_info is None:
        used = {"rs": set(), "si": set(), "ir": set()}
        accepted: List[Dict[str, Any]] = []
        for cyc in ordered:
            if any(cyc[_INDEX_KEY[pair]] in used[pair] for pair in used):
                continue
            for pair in used:
                used[pair].add(cyc[_INDEX_KEY[pair]])
            accepted.append(cyc)
        return accepted

    row_to_node = _row_to_node_maps(node_info)
    used_r: set = set()
    used_s: set = set()
    used_i: set = set()
    used_rows = {pair: set() for pair in _INDEX_KEY}
    accepted = []
    for cyc in ordered:
        rs_nodes = row_to_node["rs"].get(cyc["rs_index"])
        si_nodes = row_to_node["si"].get(cyc["si_index"])
        ir_nodes = row_to_node["ir"].get(cyc["ir_index"])
        if rs_nodes is None or si_nodes is None or ir_nodes is None:
            continue
        # Linking tolerance can exceed the node-clustering radius. Reserve both
        # endpoint estimates in each modality, and every original edge row.
        nodes_r = {rs_nodes[0], ir_nodes[1]}
        nodes_s = {rs_nodes[1], si_nodes[0]}
        nodes_i = {si_nodes[1], ir_nodes[0]}
        if (
            nodes_r & used_r
            or nodes_s & used_s
            or nodes_i & used_i
            or any(cyc[_INDEX_KEY[pair]] in used_rows[pair] for pair in used_rows)
        ):
            continue
        used_r.update(nodes_r)
        used_s.update(nodes_s)
        used_i.update(nodes_i)
        for pair in used_rows:
            used_rows[pair].add(cyc[_INDEX_KEY[pair]])
        accepted.append(cyc)
    return accepted


def _evidence_deletions(
    baseline: Dict[str, np.ndarray],
    cycles: List[Dict[str, Any]],
    edge_verifier: Optional[Callable],
    baseline_support: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Revisit the earlier pruning policy with explicit same-direction conflicts.

    Closed-cycle exteriors are removed without a feature gate only when an
    accepted edge of the same direction has a shared endpoint within 1 px
    and its other endpoint diverges by more than 4 px. Open-path competition
    keeps the 4 px divergence threshold, then keeps the top verified path and
    removes losing branches while protecting every winning edge.
    """
    pairs = ("rs", "si", "ir")
    deleted = {pair: np.zeros(len(baseline[pair]), dtype=bool) for pair in pairs}
    reasons: Dict[str, List[Dict[str, Any]]] = {pair: [] for pair in pairs}
    counts = {"protected_cycles": len(cycles), "open_candidates": 0, "selected_paths": 0}
    join_radius = _EVIDENCE_PATH_JOIN_RADIUS
    shared_radius = _EVIDENCE_CONFLICT_SHARED_RADIUS

    def incompatible(pair: str, first: int, second: int, gap: float) -> bool:
        left, right = baseline[pair][[first, second], :4]
        source_distance = np.linalg.norm(left[:2] - right[:2])
        target_distance = np.linalg.norm(left[2:4] - right[2:4])
        return bool(
            (source_distance <= shared_radius and target_distance > gap)
            or (target_distance <= shared_radius and source_distance > gap)
        )

    evidence: Dict[str, Dict[int, Tuple[bool, float]]] = {pair: {} for pair in pairs}

    def verify(indices_by_pair: Dict[str, set]) -> None:
        for pair in pairs:
            indices = sorted(indices_by_pair[pair] - evidence[pair].keys())
            if not indices:
                continue
            passed, forward_peak, reverse_peak = edge_verifier(pair, baseline[pair][indices])
            passed = np.asarray(passed, dtype=bool)
            forward_peak = np.asarray(forward_peak)
            reverse_peak = np.asarray(reverse_peak)
            if any(value.shape != (len(indices),) for value in (passed, forward_peak, reverse_peak)):
                raise ValueError("edge_verifier must return one decision and two peaks per edge")
            evidence[pair].update({
                index: (bool(passed[k]), float(min(forward_peak[k], reverse_peak[k])))
                for k, index in enumerate(indices)
            })

    cycle_rows = {
        pair: {cycle[_INDEX_KEY[pair]] for cycle in cycles} for pair in pairs
    }
    cycle_conflicts = {pair: {} for pair in pairs}
    for pair in pairs:
        cycle_indices = sorted(cycle_rows[pair])
        if not cycle_indices:
            continue
        cycle_edges = baseline[pair][cycle_indices]
        source_tree = cKDTree(cycle_edges[:, :2])
        target_tree = cKDTree(cycle_edges[:, 2:4])
        for index in range(len(baseline[pair])):
            if index in cycle_rows[pair] or baseline_support[pair][index] == 1:
                continue
            row = baseline[pair][index]
            nearby = set(source_tree.query_ball_point(row[:2], shared_radius))
            nearby.update(target_tree.query_ball_point(row[2:4], shared_radius))
            competitors = {
                cycle_indices[position] for position in nearby
                if incompatible(
                    pair, index, cycle_indices[position], _EVIDENCE_CYCLE_EXTERIOR_GAP
                )
            }
            if competitors:
                cycle_conflicts[pair][index] = competitors
    for pair in pairs:
        for index in cycle_conflicts[pair]:
            deleted[pair][index] = True
            reasons[pair].append(
                {"original_index": index, "reason": "protected_cycle_conflict"}
            )

    if edge_verifier is None:
        return deleted, reasons, counts

    # The original support/completion radius remains unchanged. Only step ③
    # considers two edges consecutive when their shared endpoint is within 0.5 px.
    paths = []
    for first, second in (("rs", "si"), ("si", "ir"), ("ir", "rs")):
        if not len(baseline[second]):
            continue
        tree = cKDTree(baseline[second][:, :2])
        for i, row in enumerate(baseline[first]):
            if deleted[first][i] or i in cycle_rows[first]:
                continue
            for j in tree.query_ball_point(row[2:4], join_radius):
                if deleted[second][j] or j in cycle_rows[second]:
                    continue
                if np.linalg.norm(row[2:4] - baseline[second][j, :2]) > join_radius:
                    continue
                paths.append((first, i, second, j))

    # A shared node between paths is ordinary connectivity. Only incompatible
    # same-direction edges make two paths competitors.
    parent = list(range(len(paths)))
    conflicts = [set() for _ in paths]

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {pair: {} for pair in pairs}
    for path_index, (first, i, second, j) in enumerate(paths):
        owners[first].setdefault(i, []).append(path_index)
        owners[second].setdefault(j, []).append(path_index)
    for pair in pairs:
        indices = sorted(owners[pair])
        if len(indices) < 2:
            continue
        rows = baseline[pair][indices]
        source_tree = cKDTree(rows[:, :2])
        target_tree = cKDTree(rows[:, 2:4])
        for position, index in enumerate(indices):
            nearby = set(source_tree.query_ball_point(rows[position, :2], shared_radius))
            nearby.update(target_tree.query_ball_point(rows[position, 2:4], shared_radius))
            for other_position in nearby:
                if other_position <= position:
                    continue
                other = indices[other_position]
                if not incompatible(pair, index, other, _EVIDENCE_PATH_CONFLICT_GAP):
                    continue
                for first_path in owners[pair][index]:
                    for second_path in owners[pair][other]:
                        conflicts[first_path].add(second_path)
                        conflicts[second_path].add(first_path)
                        parent[root(first_path)] = root(second_path)
    groups: Dict[int, List[int]] = {}
    for index in range(len(paths)):
        groups.setdefault(root(index), []).append(index)
    competing = [indices for indices in groups.values() if len(indices) > 1]
    counts["open_candidates"] = sum(len(indices) for indices in competing)
    if not competing:
        return deleted, reasons, counts

    check_indices = {pair: set() for pair in pairs}
    for indices in competing:
        for path_index in indices:
            first, i, second, j = paths[path_index]
            check_indices[first].add(i)
            check_indices[second].add(j)
    verify(check_indices)

    for indices in competing:
        valid = []
        for path_index in indices:
            first, i, second, j = paths[path_index]
            if not (evidence[first][i][0] and evidence[second][j][0]):
                continue
            confidence = float(np.sqrt(baseline[first][i, 4] * baseline[second][j, 4]))
            peak = min(evidence[first][i][1], evidence[second][j][1])
            valid.append((confidence, peak, path_index))
        if not valid:
            continue  # no reliable winner: keep the original edges
        valid.sort(key=lambda item: (-item[0], -item[1], item[2]))
        winning_edges = set()
        blocked_paths = set()
        for _, _, path_index in valid:
            if path_index in blocked_paths:
                continue
            first, i, second, j = paths[path_index]
            blocked_paths.add(path_index)
            blocked_paths.update(conflicts[path_index])
            winning_edges.update(((first, i), (second, j)))
            counts["selected_paths"] += 1
        for path_index in indices:
            first, i, second, j = paths[path_index]
            for pair, index in ((first, i), (second, j)):
                if (pair, index) in winning_edges or deleted[pair][index]:
                    continue
                deleted[pair][index] = True
                reasons[pair].append(
                    {"original_index": index, "reason": "competing_two_edge_branch"}
                )
    return deleted, reasons, counts



def _augment_topk(
    enhanced: Dict[str, np.ndarray],
    epsilon: float,
    augment_radius: float,
    min_edge_conf: float,
    link_sigma: Optional[float],
    check_target: bool,
    local_verify: bool,
    local_k: int,
    local_thresh: float,
    replace_direct: bool,
    replace_margin: float,
    topk_k: int,
    topk_lambda_geo: float,
    chain_inputs: Optional[Dict[str, np.ndarray]] = None,
    excluded_edges: Optional[Dict[str, np.ndarray]] = None,
    exclude_radius: float = 3.0,
    augmentation_stats: Optional[Dict[str, int]] = None,
    path_verifier: Optional[Callable] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Verify available paths before the best one occupies a source point."""
    chain_inputs = enhanced if chain_inputs is None else chain_inputs
    rs, si, ir = chain_inputs["rs"], chain_inputs["si"], chain_inputs["ir"]

    def reverse(matches: np.ndarray) -> np.ndarray:
        if len(matches) == 0:
            return np.empty((0, 5), dtype=np.float64)
        return np.column_stack((matches[:, 2:4], matches[:, 0:2], matches[:, 4]))

    path_inputs = {
        "rs": (reverse(ir), reverse(si)),
        "si": (reverse(rs), reverse(ir)),
        "ir": (reverse(si), reverse(rs)),
    }
    synthesized = {}
    for pair, (first, second) in path_inputs.items():
        args = (first, second, epsilon, min_edge_conf, link_sigma,
                topk_k, topk_lambda_geo)
        if path_verifier is None:
            # Low-level callers without cached feature maps retain the
            # historical single-best-path behavior.
            synthesized[pair] = transitive_chain_topk(*args)
            continue
        candidates, scores = transitive_chain_candidates(*args)
        if len(candidates):
            passed, _, _ = path_verifier(pair, candidates)
            passed = np.asarray(passed, dtype=bool)
            if passed.shape != (len(candidates),):
                raise ValueError("Path verifier must return one decision per candidate.")
            selected = np.flatnonzero(passed)
            selected = selected[np.argsort(-scores[selected], kind="stable")]
            synthesized[pair] = candidates[selected]
        else:
            synthesized[pair] = candidates
            passed = np.zeros(0, dtype=bool)
        if augmentation_stats is not None:
            augmentation_stats[f"paths_enumerated_{pair}"] = len(candidates)
            augmentation_stats[f"paths_feature_passed_{pair}"] = int(passed.sum())

    derived_mask: Dict[str, np.ndarray] = {}
    for pair in ("rs", "si", "ir"):
        existing = enhanced[pair]
        excluded = (
            excluded_edges[pair]
            if excluded_edges is not None
            else np.empty((0, 5), dtype=np.float64)
        )
        excluded_tree = cKDTree(excluded[:, :2]) if len(excluded) else None
        n_direct = len(existing)
        src_tree = cKDTree(existing[:, 0:2]) if n_direct else None
        tgt_tree = cKDTree(existing[:, 2:4]) if check_target and n_direct else None
        accepted: List[np.ndarray] = []
        added_sources: List[np.ndarray] = []
        replaced_rows: set = set()
        direct_derived = np.zeros(n_direct, dtype=bool)

        for cand in synthesized[pair]:
            source = cand[0:2]
            if excluded_tree is not None:
                nearby = excluded_tree.query_ball_point(source, exclude_radius)
                if any(
                    np.linalg.norm(cand[2:4] - excluded[index, 2:4]) <= exclude_radius
                    for index in nearby
                ):
                    if augmentation_stats is not None:
                        augmentation_stats["blocked_backfills"] += 1
                    continue
            conflict_idx = -1
            covered = False
            if src_tree is not None:
                distance, row_index = src_tree.query(source, k=1)
                if np.isfinite(distance) and distance < augment_radius:
                    covered = True
                    conflict_idx = int(row_index)
            if not covered:
                covered = any(
                    np.linalg.norm(previous - source) < augment_radius for previous in added_sources
                )

            if covered:
                if (
                    replace_direct
                    and conflict_idx >= 0
                    and conflict_idx not in replaced_rows
                    and float(cand[4]) > float(existing[conflict_idx, 4]) + replace_margin
                    and (
                        not local_verify
                        or _passes_local_geometry(cand, existing, local_k, local_thresh)
                    )
                ):
                    existing[conflict_idx] = cand
                    direct_derived[conflict_idx] = True
                    replaced_rows.add(conflict_idx)
                continue

            if tgt_tree is not None:
                target_distance, _ = tgt_tree.query(cand[2:4], k=1)
                if np.isfinite(target_distance) and target_distance < augment_radius:
                    continue
            if local_verify and not _passes_local_geometry(cand, existing, local_k, local_thresh):
                continue
            accepted.append(cand)
            added_sources.append(source)

        if accepted:
            enhanced[pair] = np.vstack((existing, np.asarray(accepted)))
        else:
            enhanced[pair] = existing
        mask = np.zeros(len(enhanced[pair]), dtype=bool)
        mask[:n_direct] = direct_derived
        mask[n_direct:] = True
        derived_mask[pair] = mask
    return enhanced, derived_mask


def refine_tri_cycle_matches(
    rs: Any,
    si: Any,
    ir: Any,
    epsilon: float = 2.0,
    tau: float = 4.0,
    sigma: float = 2.0,
    boost: float = 0.0,
    augment_radius: float = 3.0,
    do_weighting: bool = True,
    do_correction: bool = True,
    do_augmentation: bool = True,
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
    anchor: Optional[str] = "si",
    anchor_strength: float = 0.8,
    sigma_s: Optional[float] = None,
    sigma_i: Optional[float] = None,
    sigma_r: Optional[float] = None,
    top_k: int = 1,
    min_cycle_score: float = 0.0,
    one_to_one: bool = True,
    lam_si: float = 1.0,
    max_correction: float = 0.0,
    min_edge_conf: float = 0.0,
    link_sigma: Optional[float] = None,
    check_target: bool = False,
    local_verify: bool = False,
    local_k: int = 12,
    local_thresh: float = 4.0,
    replace_direct: bool = False,
    replace_margin: float = 0.0,
    route_by_support: bool = True,
    cycle_refiner: Optional[Callable] = None,
    edge_verifier: Optional[Callable] = None,
    augmentation_verifier: Optional[Callable] = None,
    do_evidence_pruning: bool = True,
    deleted_edge_radius: float = 5.0,
) -> Dict[str, Any]:
    """Refine tri-modal matches through one shared cycle pipeline.

    Original isolated edges bypass correction/completion; linked edges can
    supply two-hop completion; accepted original triangles supply correction.
    Accepted cycles protect their own edges. Step ③ removes geometrically
    conflicting cycle exteriors and losing competing two-edge branches;
    isolated edges stay intact.
    A final geometry check is applied by TCMFine, not by this low-level helper.
    ``route_by_support=False`` keeps the historical completion source pool.
    ``augmentation_verifier`` checks every eligible two-hop path before a
    source is occupied; without it the historical single-path choice remains.
    """
    rs = _validate_matches(rs, "rs")
    si = _validate_matches(si, "si")
    ir = _validate_matches(ir, "ir")
    _validate_parameters(epsilon, tau, sigma, boost, augment_radius)
    if not np.isfinite(deleted_edge_radius) or deleted_edge_radius <= 0:
        raise ValueError("deleted_edge_radius must be finite and positive")
    baseline = {"rs": rs.copy(), "si": si.copy(), "ir": ir.copy()}
    enhanced = {"rs": rs.copy(), "si": si.copy(), "ir": ir.copy()}

    node_info = None
    if use_node_clustering:
        node_info = cluster_trimodal_matches(
            baseline,
            radius_r=cluster_radius_r,
            radius_s=cluster_radius_s,
            radius_i=cluster_radius_i,
            strategy=cluster_strategy,
        )

    all_valid_cycles = _find_cycles(
        rs=rs,
        si=si,
        ir=ir,
        epsilon=epsilon,
        tau=tau,
        sigma=sigma,
        sigma_s=sigma_s,
        sigma_i=sigma_i,
        sigma_r=sigma_r,
        top_k=top_k,
        min_cycle_score=min_cycle_score,
        one_to_one=False,
    )
    if use_triangle_scoring and all_valid_cycles:
        _score_cycles(
            all_valid_cycles,
            lambda_s,
            lambda_i,
            lambda_r,
            lambda_d,
            score_min,
            score_max,
        )
    cycles = (
        _node_aware_one_to_one(all_valid_cycles, node_info if use_node_clustering else None)
        if one_to_one else all_valid_cycles
    )
    baseline_support = classify_edge_support(baseline, cycles, epsilon)
    if do_evidence_pruning:
        deleted_mask, deletion_reasons, deletion_counts = _evidence_deletions(
            baseline, cycles, edge_verifier, baseline_support,
        )
    else:
        deleted_mask = {pair: np.zeros(len(rows), dtype=bool) for pair, rows in baseline.items()}
        deletion_reasons = {pair: [] for pair in baseline}
        deletion_counts = {"protected_cycles": 0, "open_candidates": 0, "selected_paths": 0}

    original_indices = {pair: np.flatnonzero(~deleted_mask[pair]) for pair in baseline}
    working_baseline = {
        pair: baseline[pair][original_indices[pair]].copy() for pair in baseline
    }
    enhanced = {pair: enhanced[pair][original_indices[pair]].copy() for pair in baseline}
    index_maps = {
        pair: {int(original): new for new, original in enumerate(original_indices[pair])}
        for pair in baseline
    }
    working_cycles = []
    for cycle in cycles:
        if any(deleted_mask[pair][cycle[_INDEX_KEY[pair]]] for pair in baseline):
            continue
        remapped = dict(cycle)
        for pair in baseline:
            key = _INDEX_KEY[pair]
            remapped[f"original_{key}"] = cycle[key]
            remapped[key] = index_maps[pair][cycle[key]]
        working_cycles.append(remapped)
    if not any(mask.any() for mask in deleted_mask.values()):
        # Preserve the historical object identity and C's feature annotations
        # when no indices had to move.
        working_cycles = cycles
    post_prune_support = classify_edge_support(working_baseline, working_cycles, epsilon)
    if do_weighting and boost > 0.0:
        enhanced = _apply_weighting(enhanced, working_cycles, boost)
    if do_correction:
        enhanced = _apply_correction(
            enhanced,
            working_cycles,
            anchor=anchor,
            anchor_strength=anchor_strength,
            lam_si=lam_si,
            max_correction=max_correction,
        )

    cycle_feature_refinement = None
    if cycle_refiner is not None and working_cycles:
        enhanced, cycle_feature_refinement = cycle_refiner(
            enhanced, working_baseline, working_cycles, epsilon=epsilon, tau=tau
        )

    derived_mask = {pair: np.zeros(0, dtype=bool) for pair in ("rs", "si", "ir")}
    augmentation_stats = {"blocked_backfills": 0}
    if do_augmentation:
        augment = _augment_topk
        augment_kwargs = dict(
            enhanced=enhanced,
            epsilon=epsilon,
            augment_radius=augment_radius,
            min_edge_conf=min_edge_conf,
            link_sigma=link_sigma,
            check_target=check_target,
            local_verify=local_verify,
            local_k=local_k,
            local_thresh=local_thresh,
            replace_direct=replace_direct,
            replace_margin=replace_margin,
            excluded_edges={pair: baseline[pair][deleted_mask[pair]] for pair in baseline},
            exclude_radius=deleted_edge_radius,
            augmentation_stats=augmentation_stats,
            path_verifier=augmentation_verifier,
        )
        augment_kwargs.update(
            topk_k=topk_k if use_topk_augmentation else 1,
            topk_lambda_geo=topk_lambda_geo if use_topk_augmentation else 0.0,
            chain_inputs=(
                {pair: rows[post_prune_support[pair] >= 2] for pair, rows in enhanced.items()}
                if route_by_support
                else None
            ),
        )
        enhanced, derived_mask = augment(**augment_kwargs)

    for pair in ("rs", "si", "ir"):
        n_base, n_enhanced = len(working_baseline[pair]), len(enhanced[pair])
        if len(derived_mask[pair]) != n_enhanced:
            mask = np.zeros(n_enhanced, dtype=bool)
            if n_enhanced > n_base:
                mask[n_base:] = True
            derived_mask[pair] = mask

    edge_support = {}
    for pair in ("rs", "si", "ir"):
        labels = np.full(len(enhanced[pair]), 2, dtype=np.uint8)
        labels[: len(working_baseline[pair])] = post_prune_support[pair]
        labels[derived_mask[pair]] = 2
        edge_support[pair] = labels

    errors = np.asarray([cycle["cycle_error"] for cycle in working_cycles], dtype=np.float64)
    triangle_scores = np.asarray(
        [cycle.get("triangle_score", 0.0) for cycle in working_cycles],
        dtype=np.float64,
    )
    stats = {
        "n_cycles": len(working_cycles),
        "n_cycles_before_pruning": len(cycles),
        "protected_cycles": deletion_counts["protected_cycles"],
        "open_candidates": deletion_counts["open_candidates"],
        "selected_paths": deletion_counts["selected_paths"],
        "blocked_backfills": augmentation_stats["blocked_backfills"],
        "deleted_total": sum(int(mask.sum()) for mask in deleted_mask.values()),
        "cycle_error_mean": float(errors.mean()) if len(errors) else 0.0,
        "cycle_error_median": (float(np.median(errors)) if len(errors) else 0.0),
        "triangle_score_mean": (float(triangle_scores.mean()) if len(triangle_scores) else 0.0),
    }
    stats.update({
        key: int(value) for key, value in augmentation_stats.items()
        if key.startswith("paths_")
    })
    for pair in ("rs", "si", "ir"):
        stats[f"baseline_{pair}"] = len(baseline[pair])
        stats[f"enhanced_{pair}"] = len(enhanced[pair])
        stats[f"derived_{pair}"] = int(derived_mask[pair].sum())
        stats[f"deleted_{pair}"] = int(deleted_mask[pair].sum())
        stats[f"deleted_conflict_{pair}"] = sum(
            item["reason"] == "protected_cycle_conflict" for item in deletion_reasons[pair]
        )
        stats[f"deleted_open_{pair}"] = sum(
            item["reason"] == "competing_two_edge_branch" for item in deletion_reasons[pair]
        )
        for level in (1, 2, 3):
            stats[f"baseline_support_{level}_{pair}"] = int((baseline_support[pair] == level).sum())
            stats[f"post_prune_support_{level}_{pair}"] = int(
                (post_prune_support[pair] == level).sum()
            )

    result = {
        "enhanced": enhanced,
        "baseline": baseline,
        "derived_mask": derived_mask,
        "baseline_support": baseline_support,
        "post_prune_support": post_prune_support,
        "edge_support": edge_support,
        "cycles": cycles,
        "working_cycles": working_cycles,
        "original_indices": original_indices,
        "deleted_mask": deleted_mask,
        "deletion_reasons": deletion_reasons,
        "stats": stats,
    }
    if node_info is not None:
        result["node_info"] = node_info
    if cycle_feature_refinement is not None:
        result["cycle_feature_refinement"] = cycle_feature_refinement
        stats.update(
            {f"cycle_feature_{key}": value for key, value in cycle_feature_refinement.items()}
        )
    return result
