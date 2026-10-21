"""Run TCMNet inference and evaluate its three matching branches."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from configs.default import DIRECTIONS
from data.io import read_image
from evaluation import (
    build_gt_homographies,
    evaluate_direction,
    homography_inlier_mask,
    print_table,
)
from models import TCMNet
from models.checkpoint import load_pair_matcher


def add_enhance_args(parser: argparse.ArgumentParser) -> None:
    """Register the cycle-discovery / correction / augmentation options.

    Callers own the ``--epsilon``/``--tau``/``--sigma``/``--filter*`` flags.
    Final geometry is enabled unless ``--no-filter`` is set.
    """
    group = parser.add_argument_group("tri-cycle enhancement")

    # --- cycle discovery: top-K + three-residual quality + one-to-one ---
    group.add_argument(
        "--anchor",
        choices=("confidence", "si", "ir", "none"),
        default="confidence",
        help="correction policy: confidence=adaptive confidence-weighted fusion "
        "(default), si=hard 2->3 anchor (old), ir=average shared modality 2, "
        "none=symmetric mean",
    )
    group.add_argument(
        "--top-k", type=int, default=5, help="candidate links explored per hop (1 = legacy NN)"
    )
    group.add_argument(
        "--sigma-s", type=float, default=4.0, help="Gaussian scale for the modality 2 link residual"
    )
    group.add_argument(
        "--sigma-i", type=float, default=4.0, help="Gaussian scale for the modality 3 link residual"
    )
    group.add_argument(
        "--sigma-r", type=float, default=6.0, help="Gaussian scale for the modality 1 closure residual"
    )
    group.add_argument(
        "--min-cycle-score",
        type=float,
        default=0.15,
        help="reject cycle candidates with quality Q below this",
    )

    # --- adaptive correction ---
    group.add_argument(
        "--lam-si",
        type=float,
        default=1.0,
        help="2->3 branch weight in confidence fusion (1.0 = no prior, >1 trusts SI more)",
    )
    group.add_argument(
        "--max-correction",
        type=float,
        default=4.0,
        help="cap on how far (px) correction may move a point",
    )

    # --- trusted augmentation ---
    group.add_argument(
        "--min-edge-conf",
        type=float,
        default=0.0,
        help="require both chained edges to have conf >= this",
    )
    group.add_argument(
        "--link-sigma",
        type=float,
        default=None,
        help="decay derived conf by link mismatch with this scale",
    )
    group.add_argument(
        "--check-target",
        action="store_true",
        help="also reject augmented rows colliding on the target",
    )
    group.add_argument(
        "--local-verify",
        action="store_true",
        help="verify each augmented row against the local displacement field of its neighbours",
    )
    group.add_argument(
        "--local-k", type=int, default=12, help="neighbours used by local geometric verification"
    )
    group.add_argument(
        "--local-thresh",
        type=float,
        default=4.0,
        help="px tolerance for local geometric verification",
    )
    group.add_argument(
        "--replace-direct",
        action="store_true",
        help="let a clearly stronger augmented candidate replace a "
        "weaker conflicting direct match (off = never touch "
        "direct matches, only fill gaps)",
    )
    group.add_argument(
        "--replace-margin",
        type=float,
        default=0.0,
        help="confidence margin a candidate must beat to replace a "
        "direct match (with --replace-direct)",
    )

    # --- integrated cycle model ---
    group.add_argument("--cluster-radius-r", type=float, default=2.5)
    group.add_argument("--cluster-radius-s", type=float, default=2.5)
    group.add_argument("--cluster-radius-i", type=float, default=2.5)
    group.add_argument(
        "--cluster-strategy",
        choices=("highest_conf", "weighted_centroid", "median"),
        default="highest_conf",
    )
    group.add_argument("--lambda-s", type=float, default=0.4)
    group.add_argument("--lambda-i", type=float, default=0.4)
    group.add_argument("--lambda-r", type=float, default=1.0)
    group.add_argument("--lambda-d", type=float, default=2.0)
    group.add_argument("--score-min", type=float, default=-10.0)
    group.add_argument("--score-max", type=float, default=0.0)
    group.add_argument("--topk-k", type=int, default=3)
    group.add_argument("--topk-lambda-geo", type=float, default=0.1)

    # --- cycle-branch and competing two-edge pruning ---
    group.add_argument("--deleted-edge-radius", type=float, default=5.0)


def enhance_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate a parsed namespace into ``TCMFine`` keyword arguments.

    Maps ``--anchor none`` to ``anchor=None``; every other option passes through.
    """
    anchor = None if args.anchor == "none" else args.anchor
    return {
        "anchor": anchor,
        "top_k": args.top_k,
        "sigma_s": args.sigma_s,
        "sigma_i": args.sigma_i,
        "sigma_r": args.sigma_r,
        "min_cycle_score": args.min_cycle_score,
        "lam_si": args.lam_si,
        "max_correction": args.max_correction,
        "min_edge_conf": args.min_edge_conf,
        "link_sigma": args.link_sigma,
        "check_target": args.check_target,
        "local_verify": args.local_verify,
        "local_k": args.local_k,
        "local_thresh": args.local_thresh,
        "replace_direct": args.replace_direct,
        "replace_margin": args.replace_margin,
        "cluster_radius_r": args.cluster_radius_r,
        "cluster_radius_s": args.cluster_radius_s,
        "cluster_radius_i": args.cluster_radius_i,
        "cluster_strategy": args.cluster_strategy,
        "lambda_s": args.lambda_s,
        "lambda_i": args.lambda_i,
        "lambda_r": args.lambda_r,
        "lambda_d": args.lambda_d,
        "score_min": args.score_min,
        "score_max": args.score_max,
        "topk_k": args.topk_k,
        "topk_lambda_geo": args.topk_lambda_geo,
        "deleted_edge_radius": args.deleted_edge_radius,
    }


__all__ = ["add_enhance_args", "enhance_kwargs"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality1", required=True, help="modality 1 image path")
    parser.add_argument("--modality2", required=True, help="modality 2 image path")
    parser.add_argument("--modality3", required=True, help="modality 3 image path")
    parser.add_argument("--weights", default="weights/weights.ckpt")
    parser.add_argument("--image-size", type=int, choices=(512,), default=512)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="modality 2/3 link tolerance in px (stage-aware default)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="modality 1 cycle-closure tolerance in px (stage-aware default)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian scale for cycle error (stage-aware default)",
    )
    parser.add_argument(
        "--anchor-strength",
        type=float,
        default=1.0,
        help="1.0 = hard-snap weak branches onto the 2->3 anchor",
    )
    geometry_group = parser.add_mutually_exclusive_group()
    geometry_group.add_argument(
        "--no-filter",
        dest="filter",
        action="store_false",
        help="disable final MAGSAC verification (enabled by default)",
    )
    geometry_group.add_argument(
        "--filter", action="store_true", help="enable final MAGSAC verification (default)"
    )
    parser.set_defaults(filter=True)
    parser.add_argument(
        "--filter-thresh",
        type=float,
        default=3.0,
        help="pixel threshold for built-in final MAGSAC verification (default: 3)",
    )
    parser.add_argument(
        "--ransac-thresh",
        "--reproj-thresh",
        dest="ransac_thresh",
        type=float,
        default=3.0,
        help="inlier reprojection threshold in pixels for either supplied GT or MAGSAC",
    )
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--viz",
        default="outputs/tcmnet_result.png",
        help="path to save the baseline-vs-enhanced match figure",
    )
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--gt", default=None, help="path to a 3x3 ground-truth homography txt file")
    parser.add_argument(
        "--gt-shared-modality1",
        action="store_true",
        dest="gt_shared_modality1",
        help="the --gt matrix maps modality 1 to both modalities 2 and 3; "
        "derive 2->3 and 3->1 GT automatically",
    )
    parser.add_argument(
        "--gt-direction",
        choices=("rs", "si", "ir"),
        default="rs",
        help="which direction the GT homography applies to (rs=1->2, si=2->3, ir=3->1)",
    )
    add_enhance_args(parser)
    return parser.parse_args(argv)


def read_gray_tensor(path, image_size, device, half_precision):
    """Read paths containing non-ASCII characters and return [1,1,H,W]."""
    image = read_image(path)
    image = cv2.resize(image, (image_size, image_size))
    tensor = torch.from_numpy(image)[None, None].to(device=device).float() / 255.0
    return tensor.half() if half_precision else tensor


def draw_tri_modal_matches(color_images, enhanced, derived_mask, image_size, max_lines=400):
    """Draw modality 1 | modality 2 over modality 3."""
    img1, img2, img3 = (color_images[key] for key in ("modality1", "modality2", "modality3"))
    h0, w0 = img1.shape[:2]
    h1, w1 = img2.shape[:2]
    h2, w2 = img3.shape[:2]

    canvas = np.full((max(h0, h1 + h2), w0 + max(w1, w2), 3), 255, dtype=np.uint8)
    canvas[:h0, :w0] = img1
    canvas[:h1, w0 : w0 + w1] = img2
    canvas[h1 : h1 + h2, w0 : w0 + w2] = img3

    offsets = {
        "modality1": np.array([0.0, 0.0]),
        "modality2": np.array([float(w0), 0.0]),
        "modality3": np.array([float(w0), float(h1)]),
    }
    sizes = {"modality1": (w0, h0), "modality2": (w1, h1), "modality3": (w2, h2)}

    def draw_subset(matches, indices, color, src, dst):
        if len(indices) > max_lines:
            indices = np.random.default_rng(0).choice(indices, max_lines, replace=False)
        src_scale = np.array(sizes[src], dtype=np.float64) / image_size
        dst_scale = np.array(sizes[dst], dtype=np.float64) / image_size
        for i in indices:
            p0 = matches[i, :2] * src_scale + offsets[src]
            p1 = matches[i, 2:4] * dst_scale + offsets[dst]
            q0 = tuple(np.round(p0).astype(int))
            q1 = tuple(np.round(p1).astype(int))
            cv2.line(canvas, q0, q1, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, q0, 1, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, q1, 1, color, -1, cv2.LINE_AA)

    for key, src, dst in (("rs", "modality1", "modality2"),
                          ("si", "modality2", "modality3"),
                          ("ir", "modality3", "modality1")):
        matches = enhanced.get(key)
        if matches is None or len(matches) == 0:
            continue
        derived = derived_mask.get(key)
        if derived is None or len(derived) != len(matches):
            derived = np.zeros(len(matches), dtype=bool)
        inlier = homography_inlier_mask(matches, 3.0)
        draw_subset(matches, np.flatnonzero(inlier & ~derived), (70, 210, 70), src, dst)
        draw_subset(matches, np.flatnonzero(inlier & derived), (62, 225, 248), src, dst)
    return canvas


def main():
    args = parse_args()

    paths = {key: getattr(args, key) for key in ("modality1", "modality2", "modality3")}

    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{name.upper()} image does not exist: {path}. "
                f"Use --{name} to provide a valid path."
            )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    half_precision = device.type == "cuda" and not args.full_precision
    images = {
        key: read_gray_tensor(paths[key], args.image_size, device, half_precision)
        for key in ("modality1", "modality2", "modality3")
    }

    matcher = load_pair_matcher(args.weights)
    matcher = matcher.eval().to(device)
    if half_precision:
        matcher = matcher.half()

    model = TCMNet(
        matcher,
        epsilon=5.0 if args.epsilon is None else args.epsilon,
        tau=8.0 if args.tau is None else args.tau,
        sigma=5.0 if args.sigma is None else args.sigma,
        anchor_strength=args.anchor_strength,
        do_filter=args.filter,
        filter_thresh=args.filter_thresh,
        **enhance_kwargs(args),
    ).eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(images["modality1"], images["modality2"], images["modality3"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    baseline_metrics = {}
    enhanced_metrics = {}
    gt_homographies = build_gt_homographies(
        args.gt,
        args.gt_direction,
        args.gt_shared_modality1,
        paths,
        args.image_size,
    )
    if gt_homographies:
        mode = "shared 1->2/1->3" if args.gt_shared_modality1 else args.gt_direction
        filter_note = (
            "output filtering uses estimated geometry"
            if args.filter
            else "geometry filtering is disabled"
        )
        print(f"Loaded GT geometry ({mode}) for evaluation; {filter_note}:")
        for pair, H_gt in gt_homographies.items():
            print(f"{pair}:\n{H_gt}")
        print()

    for label, pair in DIRECTIONS.items():
        H_gt = gt_homographies.get(pair)
        baseline_metrics[label] = evaluate_direction(
            result["baseline"][pair], args.ransac_thresh, H_gt
        )
        enhanced_metrics[label] = evaluate_direction(
            result["enhanced"][pair], args.ransac_thresh, H_gt
        )

    print("Inputs: " + ", ".join(f"{key}={path}" for key, path in paths.items()))
    print(f"Enhancer stats: {result['stats']}")
    for pair, info in result.get("geometry", {}).items():
        feature_removed = result["stats"].get(f"feature_rejected_{pair}", 0)
        geometry_removed = result["stats"][f"removed_{pair}"] - feature_removed
        print(
            f"Geometry {pair}: {info['status']}; removed_geometry={geometry_removed}; "
            f"removed_feature={feature_removed}"
        )
    print(f"Inference time: {elapsed:.4f} s\n")
    print_table(
        baseline_metrics,
        enhanced_metrics,
        gt_pair=args.gt_direction if args.gt and not args.gt_shared_modality1 else None,
    )

    if args.viz and not args.no_viz:
        color_images = {
            key: read_image(paths[key], cv2.IMREAD_COLOR)
            for key in ("modality1", "modality2", "modality3")
        }
        figure = draw_tri_modal_matches(
            color_images,
            result["enhanced"],
            result["derived_mask"],
            args.image_size,
        )
        ok, buffer = cv2.imencode(".png", figure)
        if not ok:
            raise RuntimeError("Failed to encode visualization.")
        Path(args.viz).parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(args.viz)  # non-ASCII safe write
        print(f"\nSaved visualization to {args.viz}")


if __name__ == "__main__":
    main()
