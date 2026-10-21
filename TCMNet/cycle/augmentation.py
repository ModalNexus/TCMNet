"""Top-K transitive path synthesis for two-hop augmentation.

The Top-K helper chooses one path per first edge. The candidate helper exposes
all nearby paths so cached bidirectional features can reject bad paths before
the best surviving path occupies a source point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError("topk_chain requires SciPy. Install it with: pip install scipy") from exc


def _geometric_mean2(a: float, b: float) -> float:
    """Two-value geometric mean (0 if either is 0)."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(np.sqrt(a * b))


def _query_neighbors_kdtree(
    tree: cKDTree, point: np.ndarray, k: int, radius: float
) -> List[Tuple[float, int]]:
    """Return up to k neighbors strictly within radius, sorted by distance."""
    dist, idx = tree.query(point, k=k, distance_upper_bound=radius)
    dist = np.atleast_1d(dist)
    idx = np.atleast_1d(idx)
    out: List[Tuple[float, int]] = []
    for d, j in zip(dist, idx):
        if np.isfinite(d) and int(j) < tree.n and d < radius:
            out.append((float(d), int(j)))
    return out


def transitive_chain_candidates(
    first: np.ndarray,
    second: np.ndarray,
    link_eps: float,
    min_edge_conf: float = 0.0,
    link_sigma: Optional[float] = None,
    top_k: int = 3,
    lambda_geo: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return every Top-K two-edge path and its score before source deduplication.

    The caller can verify all paths before one of them occupies a source point.
    ``top_k`` limits the nearest links examined for each first edge, while
    ``link_eps`` is a strict upper bound in the shared modality's coordinates.
    """
    if len(first) == 0 or len(second) == 0:
        return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    link_tree = cKDTree(second[:, :2])
    rows: List[np.ndarray] = []
    scores: List[float] = []
    for edge in first:
        if edge[4] < min_edge_conf:
            continue
        for distance, index in _query_neighbors_kdtree(
            link_tree, edge[2:4], top_k, link_eps
        ):
            next_edge = second[index]
            if next_edge[4] < min_edge_conf:
                continue
            confidence = _geometric_mean2(float(edge[4]), float(next_edge[4]))
            if link_sigma is not None and link_sigma > 0.0:
                confidence *= float(np.exp(-distance**2 / (2.0 * link_sigma**2)))
            score = (
                np.log(float(edge[4]) + 1e-9)
                + np.log(float(next_edge[4]) + 1e-9)
                - lambda_geo * distance**2
            )
            rows.append(np.array([
                edge[0], edge[1], next_edge[2], next_edge[3], confidence
            ]))
            scores.append(float(score))
    if not rows:
        return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def transitive_chain_topk(
    first: np.ndarray,
    second: np.ndarray,
    link_eps: float,
    min_edge_conf: float = 0.0,
    link_sigma: Optional[float] = None,
    top_k: int = 3,
    lambda_geo: float = 0.0,
) -> np.ndarray:
    """Synthesize source->target matches by chaining first->link->target.

    For each row in `first`, find the Top-K rows in `second` whose source
    coordinate (columns 0:2) is within `link_eps` of `first`'s target (columns 2:4).
    Score all K paths and return the best one.

    Path score:
        S = log(c_first) + log(c_second) - lambda_geo * d_link^2

    Args:
        first: (N, 5) [x0, y0, x1, y1, conf] array (first edge)
        second: (M, 5) [x0, y0, x1, y1, conf] array (second edge)
        link_eps: maximum distance between first's target and second's source
        min_edge_conf: minimum confidence threshold for both edges
        link_sigma: if set, apply Gaussian decay to derived confidence
        top_k: number of candidate paths to consider per first-row
        lambda_geo: geometric penalty weight (penalizes link distance)

    Returns:
        (K, 5) array of synthesized matches [x0_first, y0_first, x1_second, y1_second, conf_derived]
    """
    if len(first) == 0 or len(second) == 0:
        return np.empty((0, 5), dtype=np.float64)

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    # Build KD-tree on second's source coordinates
    link_tree = cKDTree(second[:, 0:2])

    out: List[np.ndarray] = []

    for row in first:
        if float(row[4]) < min_edge_conf:
            continue

        link = row[2:4]  # target of first edge = source of second edge
        neighbors = _query_neighbors_kdtree(link_tree, link, k=top_k, radius=link_eps)

        if not neighbors:
            continue

        best_score = -np.inf
        best_match = None

        for dist, idx in neighbors:
            srow = second[int(idx)]
            if float(srow[4]) < min_edge_conf:
                continue

            # Compute derived confidence
            conf = _geometric_mean2(float(row[4]), float(srow[4]))
            if link_sigma is not None and link_sigma > 0.0:
                conf *= float(np.exp(-(dist**2) / (2.0 * float(link_sigma) ** 2)))

            # Compute path score (log-space + geometric penalty)
            epsilon = 1e-9
            log_conf = np.log(float(row[4]) + epsilon) + np.log(float(srow[4]) + epsilon)
            geo_penalty = lambda_geo * dist**2
            score = log_conf - geo_penalty

            if score > best_score:
                best_score = score
                best_match = np.array([row[0], row[1], srow[2], srow[3], conf])

        if best_match is not None:
            out.append(best_match)

    if not out:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)
