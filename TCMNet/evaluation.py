"""Shared image-match evaluation and visualization."""

from typing import Optional, Tuple

import cv2
import numpy as np

from configs.default import DIRECTIONS
from data.io import read_image


def read_gt_homography(path):
    """Read a 3x3 ground-truth homography matrix from a txt file."""
    H = np.load(path) if str(path).lower().endswith(".npy") else np.loadtxt(path, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix in {path}, got {H.shape}")
    return H


def read_image_hw(path):
    """Return an image's original (height, width), including non-ASCII paths."""
    return read_image(path).shape[:2]


def resize_homography(H, src_path, dst_path, image_size):
    """Convert an original-image homography to resized inference coordinates."""
    src_h, src_w = read_image_hw(src_path)
    dst_h, dst_w = read_image_hw(dst_path)
    S_src = np.diag([image_size / src_w, image_size / src_h, 1.0])
    S_dst = np.diag([image_size / dst_w, image_size / dst_h, 1.0])
    H_resized = S_dst @ H @ np.linalg.inv(S_src)
    if abs(H_resized[2, 2]) > 1e-12:
        H_resized = H_resized / H_resized[2, 2]
    return H_resized


def build_gt_homographies(gt_path, gt_direction, shared_modality1, paths, image_size):
    """Build resized GT matrices for the inference direction keys.

    With ``shared_modality1``, the same original matrix maps modality 1 to
    both modalities 2 and 3. The 2→3 and 3→1 matrices are derived algebraically.
    """
    if not gt_path:
        if shared_modality1:
            raise ValueError("--gt-shared-modality1 requires --gt")
        return {}

    H = read_gt_homography(gt_path)
    if shared_modality1:
        H_12 = resize_homography(H, paths["modality1"], paths["modality2"], image_size)
        H_13 = resize_homography(H, paths["modality1"], paths["modality3"], image_size)
        H_23 = H_13 @ np.linalg.inv(H_12)
        H_31 = np.linalg.inv(H_13)
        return {"rs": H_12, "si": H_23, "ir": H_31}

    frame_paths = {
        "rs": (paths["modality1"], paths["modality2"]),
        "si": (paths["modality2"], paths["modality3"]),
        "ir": (paths["modality3"], paths["modality1"]),
    }
    src_path, dst_path = frame_paths[gt_direction]
    return {gt_direction: resize_homography(H, src_path, dst_path, image_size)}


def gt_reprojection_errors(matches, H):
    """Return one GT reprojection error per match."""
    n = len(matches)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    src = matches[:, 0:2].astype(np.float64)
    dst = matches[:, 2:4].astype(np.float64)
    src_h = np.hstack((src, np.ones((n, 1), dtype=np.float64)))
    proj_h = (H @ src_h.T).T
    valid = np.abs(proj_h[:, 2]) > 1e-12
    errors = np.full(n, np.inf, dtype=np.float64)
    proj = proj_h[valid, 0:2] / proj_h[valid, 2:3]
    errors[valid] = np.linalg.norm(proj - dst[valid], axis=1)
    return errors


def gt_inlier_mask(matches, H, reproj_thresh):
    """Return GT inliers without estimating a homography from the matches."""
    return gt_reprojection_errors(matches, H) <= reproj_thresh


def homography_inlier_mask(matches, ransac_thresh):
    """Return a bool inlier mask aligned with ``matches`` rows (False if N<4)."""
    n = len(matches)
    mask = np.zeros(n, dtype=bool)
    if n < 4:
        return mask
    src = matches[:, 0:2].astype(np.float64)
    dst = matches[:, 2:4].astype(np.float64)
    H, m = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh)
    if H is None or m is None:
        return mask
    return m.ravel().astype(bool)


def evaluate_direction(matches, ransac_thresh, gt_H=None):
    """Fit a MAGSAC homography and report inlier metrics for one direction.

    ``matches`` is an (N, 5) array ``[x0, y0, x1, y1, conf]``. The homography
    absorbs any flip/crop/scale, so this works without assuming alignment.

    If ``gt_H`` (3x3) is provided, use it directly instead of fitting MAGSAC.
    """
    n = len(matches)
    empty = {
        "matches": n,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reproj_error": float("nan"),
        "gt_inliers": 0,
        "gt_inlier_ratio": 0.0,
        "gt_reproj_error": float("nan"),
    }
    # A supplied GT matrix is the primary geometry source. Do not estimate a
    # replacement homography with MAGSAC in this branch.
    if gt_H is not None:
        errors = gt_reprojection_errors(matches, gt_H)
        gt_mask = errors <= ransac_thresh
        gt_inliers = int(gt_mask.sum())
        gt_reproj = float(errors[gt_mask].mean()) if gt_inliers else float("nan")
        return {
            "matches": n,
            "inliers": gt_inliers,
            "inlier_ratio": gt_inliers / n if n else 0.0,
            "reproj_error": gt_reproj,
            "gt_inliers": gt_inliers,
            "gt_inlier_ratio": gt_inliers / n if n else 0.0,
            "gt_reproj_error": gt_reproj,
        }
    if n < 4:
        return empty
    src = matches[:, 0:2].astype(np.float64)
    dst = matches[:, 2:4].astype(np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh)
    if H is None or mask is None:
        return empty
    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())

    # Mean reprojection error over inliers only (MAGSAC-estimated H).
    reproj = float(gt_reprojection_errors(matches[mask], H).mean()) if inliers else float("nan")

    return {
        "matches": n,
        "inliers": inliers,
        "inlier_ratio": inliers / n,
        "reproj_error": reproj,
        "gt_inliers": 0,
        "gt_inlier_ratio": 0.0,
        "gt_reproj_error": float("nan"),
    }


def print_table(baseline, enhanced, gt_pair=None):
    """Print a side-by-side baseline vs enhanced metric table.

    If ``gt_pair`` (e.g. "rs") is given, an extra GT-based evaluation section
    is appended for that direction, reporting inliers against the ground-truth
    homography.
    """
    header = (
        f"{'direction':<10} | {'set':<8} | {'matches':>7} | {'inliers':>7} | "
        f"{'ratio':>6} | {'reproj':>7}"
    )
    print(header)
    print("-" * len(header))
    for label in DIRECTIONS:
        for set_name, table in (("baseline", baseline), ("enhanced", enhanced)):
            m = table[label]
            reproj = "  nan" if np.isnan(m["reproj_error"]) else f"{m['reproj_error']:.3f}"
            print(
                f"{label:<10} | {set_name:<8} | {m['matches']:>7d} | "
                f"{m['inliers']:>7d} | {m['inlier_ratio']:>6.3f} | {reproj:>7}"
            )
        print("-" * len(header))

    if gt_pair is None:
        return

    # --- GT evaluation section ---
    gt_label = {v: k for k, v in DIRECTIONS.items()}[gt_pair]
    print(f"\nGround-Truth Evaluation  [{gt_label}]")
    gt_header = (
        f"{'set':<8} | {'matches':>7} | {'GT inliers':>10} | {'GT ratio':>8} | {'GT reproj':>8}"
    )
    print(gt_header)
    print("-" * len(gt_header))
    for set_name, table in (("baseline", baseline), ("enhanced", enhanced)):
        m = table[gt_label]
        gt_reproj = "    nan" if np.isnan(m["gt_reproj_error"]) else f"{m['gt_reproj_error']:.3f}"
        print(
            f"{set_name:<8} | {m['matches']:>7d} | {m['gt_inliers']:>10d} | "
            f"{m['gt_inlier_ratio']:>8.3f} | {gt_reproj:>8}"
        )
    print("-" * len(gt_header))


def read_color_bgr(path, image_size):
    """Read an image (non-ASCII safe) as BGR resized to image_size for drawing."""
    image = read_image(path, cv2.IMREAD_COLOR)
    # Homography-generated samples use pure-black, border-connected regions as
    # padding. Replace only those background components for a white figure;
    # genuine dark content enclosed inside the image is preserved.
    black = np.all(image == 0, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(black, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size:
        image[np.isin(labels, border_labels)] = 255
    return cv2.resize(image, (image_size, image_size))


def draw_pair_matches(
    src_img, dst_img, matches, inlier_mask, derived_mask, max_lines=400, draw_outliers=True
):
    """Draw matches between two side-by-side images.

    Inliers are green, outliers thin red, cycle-derived inliers yellow. Lines are
    subsampled to ``max_lines`` per category to keep the figure readable.
    """
    h0, w0 = src_img.shape[:2]
    h1, w1 = dst_img.shape[:2]
    canvas = np.full((max(h0, h1), w0 + w1, 3), 255, dtype=np.uint8)
    canvas[:h0, :w0] = src_img
    canvas[:h1, w0 : w0 + w1] = dst_img
    offset = np.array([w0, 0])

    derived_mask = derived_mask if derived_mask is not None else np.zeros(len(matches), dtype=bool)

    def draw_subset(idx, color, thickness, radius):
        if len(idx) > max_lines:
            idx = np.random.default_rng(0).choice(idx, max_lines, replace=False)
        for i in idx:
            p0 = tuple(np.round(matches[i, 0:2]).astype(int))
            p1 = tuple(np.round(matches[i, 2:4] + offset).astype(int))
            cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, p0, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, radius, color, -1, cv2.LINE_AA)

    all_idx = np.arange(len(matches))
    outliers = all_idx[~inlier_mask]
    inliers_direct = all_idx[inlier_mask & ~derived_mask]
    inliers_derived = all_idx[inlier_mask & derived_mask]
    # OpenCV colors are BGR. Draw outliers first when requested, then direct
    # inliers (green), then cycle-derived inliers with the requested exact
    # BGR color (62, 225, 248).
    if draw_outliers:
        draw_subset(outliers, (60, 60, 200), 1, 1)
    draw_subset(inliers_direct, (80, 220, 80), 1, 2)
    draw_subset(inliers_derived, (62, 225, 248), 1, 2)
    return canvas


def label_bar(width, text, height=28):
    """Make a white caption strip with black text."""
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def build_visualization(result, color_images, ransac_thresh, image_size, gt_homographies=None):
    """Compose a baseline-vs-enhanced match figure for all three directions.

    Layout: three direction blocks stacked vertically; each block has a
    baseline row and an enhanced row side by side (source | target).
    """
    frame_of = {  # which color image is the source / target of each direction
        "1->2": ("modality1", "modality2"),
        "2->3": ("modality2", "modality3"),
        "3->1": ("modality3", "modality1"),
    }
    gt_homographies = gt_homographies or {}
    blocks = []
    for label, pair in DIRECTIONS.items():
        src_name, dst_name = frame_of[label]
        src_img = color_images[src_name]
        dst_img = color_images[dst_name]
        rows = []
        for set_name, key in (("baseline", "baseline"), ("enhanced", "enhanced")):
            matches = result[key][pair]
            if pair in gt_homographies:
                inlier = gt_inlier_mask(matches, gt_homographies[pair], ransac_thresh)
                geometry_label = "GT"
            else:
                inlier = homography_inlier_mask(matches, ransac_thresh)
                geometry_label = "MAGSAC"
            derived = result["derived_mask"][pair] if key == "enhanced" else None
            panel = draw_pair_matches(
                src_img,
                dst_img,
                matches,
                inlier,
                derived,
                draw_outliers=True,
            )
            caption = (
                f"{label}  [{set_name}]  matches={len(matches)}  "
                f"inliers={int(inlier.sum())}  geometry={geometry_label}"
            )
            bar = label_bar(panel.shape[1], caption)
            rows.append(np.vstack((bar, panel)))
        blocks.append(np.vstack(rows))
        # Spacer between direction blocks.
        blocks.append(np.full((12, blocks[-1].shape[1], 3), 255, dtype=np.uint8))
    return np.vstack(blocks[:-1])


def homography_inliers(
    points0: np.ndarray,
    points1: np.ndarray,
    threshold: float,
    homography: Optional[np.ndarray],
) -> Tuple[np.ndarray, float, str]:
    count = len(points0)
    if count == 0:
        return np.zeros(0, dtype=bool), float("nan"), "none"

    if homography is None:
        if count < 4:
            return np.zeros(count, dtype=bool), float("nan"), "insufficient"
        homography, mask = cv2.findHomography(
            points0.astype(np.float64),
            points1.astype(np.float64),
            cv2.USAC_MAGSAC,
            threshold,
        )
        if homography is None or mask is None:
            return np.zeros(count, dtype=bool), float("nan"), "MAGSAC failed"
        inliers = mask.ravel().astype(bool)
        label = "MAGSAC"
    else:
        inliers = np.ones(count, dtype=bool)
        label = "ground truth"

    errors = gt_reprojection_errors(np.column_stack((points0, points1)), homography)
    if label == "ground truth":
        inliers = errors < threshold
    mean_error = float(errors[inliers].mean()) if inliers.any() else float("nan")
    return inliers, mean_error, label


def draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inliers: np.ndarray,
) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    canvas = np.zeros((height, image0.shape[1] + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], : image0.shape[1]] = image0
    canvas[: image1.shape[0], image0.shape[1] :] = image1
    offset = image0.shape[1]

    for index, (point0, point1) in enumerate(zip(points0, points1)):
        color = (70, 210, 70) if inliers[index] else (60, 60, 220)
        p0 = tuple(np.rint(point0).astype(int))
        p1 = tuple(np.rint(point1).astype(int) + np.array([offset, 0]))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    return canvas
