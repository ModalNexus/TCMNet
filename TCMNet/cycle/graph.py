"""Physical-node clustering and joint triangle scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError("node_clustering requires SciPy. Install it with: pip install scipy") from exc


def cluster_points(
    coords: np.ndarray,
    confidences: np.ndarray,
    radius: float,
    strategy: str = "highest_conf",
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Cluster nearby points into physical nodes.

    Args:
        coords: (N, 2) array of [x, y] coordinates
        confidences: (N,) array of confidence scores
        radius: clustering radius in pixels
        strategy: how to pick the representative coordinate
            - "highest_conf": use the highest-confidence point
            - "weighted_centroid": confidence-weighted mean
            - "median": coordinate-wise median (robust to outliers)

    Returns:
        node_coords: (M, 2) representative coordinates for M nodes
        row_to_node: (N,) mapping from original row index to node ID
        node_members: list of M lists, each containing row indices in that node
    """
    if len(coords) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            [],
        )

    coords = np.asarray(coords, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords.shape}")
    if len(confidences) != len(coords):
        raise ValueError(f"confidences length {len(confidences)} != coords length {len(coords)}")

    N = len(coords)
    visited = np.zeros(N, dtype=bool)
    row_to_node = np.full(N, -1, dtype=np.int32)
    node_coords_list: List[np.ndarray] = []
    node_members: List[List[int]] = []

    tree = cKDTree(coords)

    for i in range(N):
        if visited[i]:
            continue

        # Find all points within radius
        indices = tree.query_ball_point(coords[i], radius)
        indices = [idx for idx in indices if not visited[idx]]

        if len(indices) == 0:
            continue

        # Mark as visited
        for idx in indices:
            visited[idx] = True
            row_to_node[idx] = len(node_coords_list)

        # Compute representative coordinate
        cluster_coords = coords[indices]
        cluster_confs = confidences[indices]

        if strategy == "highest_conf":
            best_idx = indices[np.argmax(cluster_confs)]
            rep_coord = coords[best_idx].copy()
        elif strategy == "weighted_centroid":
            weights = cluster_confs / cluster_confs.sum()
            rep_coord = (cluster_coords * weights[:, None]).sum(axis=0)
        elif strategy == "median":
            rep_coord = np.median(cluster_coords, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        node_coords_list.append(rep_coord)
        node_members.append(indices)

    node_coords = (
        np.array(node_coords_list, dtype=np.float64)
        if node_coords_list
        else np.empty((0, 2), dtype=np.float64)
    )

    return node_coords, row_to_node, node_members


def cluster_trimodal_matches(
    matches: Dict[str, np.ndarray],
    radius_r: float,
    radius_s: float,
    radius_i: float,
    strategy: str = "highest_conf",
) -> Dict[str, Any]:
    """Cluster all three modalities and build row->node mappings.

    Args:
        matches: dict with keys "rs", "si", "ir", each (N, 5) [x0,y0,x1,y1,conf]
        radius_r: clustering radius for RGB modality (pixels)
        radius_s: clustering radius for SAR modality (pixels)
        radius_i: clustering radius for IR modality (pixels)
        strategy: clustering strategy (see cluster_points)

    Returns:
        Dictionary containing:
            - nodes_r, nodes_s, nodes_i: (M, 2) node coordinates
            - row_to_node_r, row_to_node_s, row_to_node_i: row->node mappings per modality
            - members_r, members_s, members_i: list of row indices per node
            - edge_nodes: dict mapping "rs"/"si"/"ir" to (N, 3) [node0, node1, row_idx]
    """
    # Extract all coordinates per modality
    all_r_coords: List[np.ndarray] = []
    all_r_confs: List[float] = []
    all_s_coords: List[np.ndarray] = []
    all_s_confs: List[float] = []
    all_i_coords: List[np.ndarray] = []
    all_i_confs: List[float] = []

    rs = matches.get("rs", np.empty((0, 5)))
    si = matches.get("si", np.empty((0, 5)))
    ir = matches.get("ir", np.empty((0, 5)))

    # RGB from RS and IR
    if len(rs) > 0:
        all_r_coords.append(rs[:, 0:2])
        all_r_confs.extend(rs[:, 4])
    if len(ir) > 0:
        all_r_coords.append(ir[:, 2:4])
        all_r_confs.extend(ir[:, 4])

    # SAR from RS and SI
    if len(rs) > 0:
        all_s_coords.append(rs[:, 2:4])
        all_s_confs.extend(rs[:, 4])
    if len(si) > 0:
        all_s_coords.append(si[:, 0:2])
        all_s_confs.extend(si[:, 4])

    # IR from SI and IR
    if len(si) > 0:
        all_i_coords.append(si[:, 2:4])
        all_i_confs.extend(si[:, 4])
    if len(ir) > 0:
        all_i_coords.append(ir[:, 0:2])
        all_i_confs.extend(ir[:, 4])

    r_coords = np.vstack(all_r_coords) if all_r_coords else np.empty((0, 2))
    s_coords = np.vstack(all_s_coords) if all_s_coords else np.empty((0, 2))
    i_coords = np.vstack(all_i_coords) if all_i_coords else np.empty((0, 2))

    r_confs = np.array(all_r_confs, dtype=np.float64)
    s_confs = np.array(all_s_confs, dtype=np.float64)
    i_confs = np.array(all_i_confs, dtype=np.float64)

    # Cluster each modality
    nodes_r, row_to_node_r_flat, members_r = cluster_points(r_coords, r_confs, radius_r, strategy)
    nodes_s, row_to_node_s_flat, members_s = cluster_points(s_coords, s_confs, radius_s, strategy)
    nodes_i, row_to_node_i_flat, members_i = cluster_points(i_coords, i_confs, radius_i, strategy)

    # Build per-edge node mappings
    # RS: rows 0..len(rs)-1 map to RGB, rows 0..len(rs)-1 map to SAR
    # SI: rows 0..len(si)-1 map to SAR (offset by len(rs)), IR (offset 0)
    # IR: rows 0..len(ir)-1 map to IR (offset by len(si)), RGB (offset by len(rs))

    rs_r_offset = 0
    rs_s_offset = 0
    si_s_offset = len(rs)
    si_i_offset = 0
    ir_i_offset = len(si)
    ir_r_offset = len(rs)

    edge_nodes = {}

    # RS edges
    if len(rs) > 0:
        rs_nodes = np.column_stack(
            [
                row_to_node_r_flat[rs_r_offset : rs_r_offset + len(rs)],
                row_to_node_s_flat[rs_s_offset : rs_s_offset + len(rs)],
                np.arange(len(rs), dtype=np.int32),
            ]
        )
        edge_nodes["rs"] = rs_nodes
    else:
        edge_nodes["rs"] = np.empty((0, 3), dtype=np.int32)

    # SI edges
    if len(si) > 0:
        si_nodes = np.column_stack(
            [
                row_to_node_s_flat[si_s_offset : si_s_offset + len(si)],
                row_to_node_i_flat[si_i_offset : si_i_offset + len(si)],
                np.arange(len(si), dtype=np.int32),
            ]
        )
        edge_nodes["si"] = si_nodes
    else:
        edge_nodes["si"] = np.empty((0, 3), dtype=np.int32)

    # IR edges
    if len(ir) > 0:
        ir_nodes = np.column_stack(
            [
                row_to_node_i_flat[ir_i_offset : ir_i_offset + len(ir)],
                row_to_node_r_flat[ir_r_offset : ir_r_offset + len(ir)],
                np.arange(len(ir), dtype=np.int32),
            ]
        )
        edge_nodes["ir"] = ir_nodes
    else:
        edge_nodes["ir"] = np.empty((0, 3), dtype=np.int32)

    return {
        "nodes_r": nodes_r,
        "nodes_s": nodes_s,
        "nodes_i": nodes_i,
        "members_r": members_r,
        "members_s": members_s,
        "members_i": members_i,
        "edge_nodes": edge_nodes,
    }


def classify_edge_support(
    matches: Dict[str, np.ndarray], cycles: List[Dict[str, Any]], link_radius: float
) -> Dict[str, np.ndarray]:
    """Label original edges: 1 isolated, 2 linked, 3 accepted triangle member.

    These are local edge roles, not sizes of connected components. A row can
    appear in several candidate paths; accepted triangles take precedence.
    Classification uses original coordinates, before correction/completion.
    A synthesized third edge never supplies independent triangle evidence.
    """
    support = {pair: np.ones(len(rows), dtype=np.uint8) for pair, rows in matches.items()}
    for first, second in (("rs", "si"), ("si", "ir"), ("ir", "rs")):
        left, right = matches[first][:, 2:4], matches[second][:, :2]
        if not len(left) or not len(right):
            continue
        distances, _ = cKDTree(right).query(left, distance_upper_bound=link_radius)
        support[first][distances < link_radius] = 2
        distances, _ = cKDTree(left).query(right, distance_upper_bound=link_radius)
        support[second][distances < link_radius] = 2
    for cycle in cycles:
        for pair in support:
            support[pair][cycle[f"{pair}_index"]] = 3
    return support


def triangle_score(
    conf_rs: float,
    conf_si: float,
    conf_ir: float,
    d_s: float,
    d_i: float,
    d_r: float,
    lambda_s: float,
    lambda_i: float,
    lambda_r: float,
    lambda_d: float,
    n_derived: int,
    epsilon: float = 1e-9,
) -> float:
    """Compute joint score for a triangle (r, s, i).

    Score = Σlog(conf + ε) - λ_s·d_s² - λ_i·d_i² - λ_r·d_r² - λ_d·N_derived

    Args:
        conf_rs, conf_si, conf_ir: edge confidences [0, 1]
        d_s: SAR coordinate residual (distance between two SAR estimates)
        d_i: IR coordinate residual
        d_r: RGB loop-closure residual
        lambda_s, lambda_i, lambda_r: residual penalty weights
        lambda_d: derived edge penalty per edge
        n_derived: number of derived (augmented) edges in this triangle
        epsilon: numerical stability constant

    Returns:
        Triangle score (higher is better)
    """
    log_conf = np.log(conf_rs + epsilon) + np.log(conf_si + epsilon) + np.log(conf_ir + epsilon)
    residual_penalty = lambda_s * d_s**2 + lambda_i * d_i**2 + lambda_r * d_r**2
    derived_penalty = lambda_d * n_derived

    return float(log_conf - residual_penalty - derived_penalty)


def cycle_weight_from_score(
    score: float,
    score_min: float = -10.0,
    score_max: float = 0.0,
) -> float:
    """Map triangle score to [0, 1] weight for backward compatibility.

    Args:
        score: triangle score (can be negative)
        score_min: score value that maps to weight 0
        score_max: score value that maps to weight 1

    Returns:
        Weight in [0, 1]
    """
    if score <= score_min:
        return 0.0
    if score >= score_max:
        return 1.0
    return float((score - score_min) / (score_max - score_min))
