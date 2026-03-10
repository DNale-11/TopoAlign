"""Compare fixed vs moving images before and after existing registration pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from skimage.transform import warp

from .config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .io_utils import infer_image_mode, load_image, project_intensity_max, save_mask
from .matching import MatchingConfig, greedy_match_cells, match_cells_per_cluster, match_cells_per_patch
from .registration import (
    build_inverse_affine_transform,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
)
from .segmentation import CellposeSegmenter
from .main import (
    DEFAULT_POSITION_WEIGHT,
    DEFAULT_RESIDUAL_PRUNE_QUANTILE,
    DEFAULT_TOP_K,
    MIN_MATCHES_FOR_REFINEMENT,
    TOP_K_PER_PATCH,
    assign_clusters_from_round1,
    assign_patches,
    compute_match_residuals,
    estimate_rigid_transform_from_matches_ransac,
    run_topology_matching_df,
)


def prepare_metric_image(image: np.ndarray) -> np.ndarray:
    """Prepare images for metrics (project Z-stacks to 2D when needed)."""
    if infer_image_mode(image) == "3d_zstack":
        return project_intensity_max(image)
    return image


def warp_image_to_fixed(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    output_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Warp moving image into fixed space using the estimated rigid transform."""
    affine = build_inverse_affine_transform(rotation, translation)
    return warp(
        image.astype(np.float32),
        inverse_map=affine,
        order=order,
        preserve_range=True,
        output_shape=output_shape,
    )

def compute_similarity_metrics(fixed_img: np.ndarray, moving_img: np.ndarray) -> dict:
    """Compute SSIM and NRMSE for two aligned images."""
    fixed = fixed_img.astype(np.float32)
    moving = moving_img.astype(np.float32)
    data_range = float(fixed.max() - fixed.min())
    if data_range <= 0:
        data_range = 1.0
    channel_axis = -1 if fixed.ndim == 3 else None
    ssim = structural_similarity(fixed, moving, data_range=data_range, channel_axis=channel_axis)
    rmse = float(np.sqrt(np.mean((fixed - moving) ** 2)))
    nrmse = rmse / data_range if data_range > 0 else 0.0
    return {"ssim": float(ssim), "nrmse": float(nrmse)}


def save_float_image(path: Path, image: np.ndarray) -> None:
    """Save float images to disk, normalizing for 8-bit formats."""
    fmt = path.suffix.lower().lstrip(".")
    if fmt in {"png", "jpg", "jpeg"}:
        img = image.astype(np.float32)
        vmin = float(np.min(img))
        vmax = float(np.max(img))
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        iio.imwrite(path, img)
    else:
        iio.imwrite(path, image.astype(np.float32))


def run_compare(
    fixed_path: Path,
    moving_path: Path,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    top_k_per_patch: int | None = TOP_K_PER_PATCH,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    use_spatial_clusters: bool = False,
    n_clusters: int = 9,
    use_topology_filtering: bool = True,
    k_pos_nei: int = 5,
    k_neighbor: int = 5,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    tau_map: float = 0.3,
    use_ransac_transform: bool = False,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    residual_prune_quantile: float | None = DEFAULT_RESIDUAL_PRUNE_QUANTILE,
    use_gpu: bool = False,
    output_format: str = "tif",
    mask_only: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if top_k_per_patch is not None and top_k_per_patch <= 0:
        top_k_per_patch = None

    fixed_img = load_image(fixed_path)
    moving_img = load_image(moving_path)

    mode1 = infer_image_mode(fixed_img)
    mode2 = infer_image_mode(moving_img)
    if mode1 != mode2:
        raise ValueError(
            f"Both images must be either 2D or 3D Z-stacks; got {mode1} and {mode2}."
        )

    cellpose_config = DEFAULT_CELLPOSE_CONFIG
    if use_gpu:
        cellpose_config = replace(DEFAULT_CELLPOSE_CONFIG, gpu=True)
    segmenter = CellposeSegmenter(cellpose_config)

    use_zstack = mode1 == "3d_zstack"
    if use_zstack:
        _masks1_3d, masks1, _, _ = segmenter.segment_zstack(fixed_img)
        _masks2_3d, masks2, _, _ = segmenter.segment_zstack(moving_img)
        fixed_metric = project_intensity_max(fixed_img)
        moving_metric = project_intensity_max(moving_img)
    else:
        masks1, _, _ = segmenter.segment_array(fixed_img)
        masks2, _, _ = segmenter.segment_array(moving_img)
        fixed_metric = fixed_img
        moving_metric = moving_img

    feats1 = compute_cell_features(masks1, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
    feats2 = compute_cell_features(masks2, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
    h1, w1 = masks1.shape

    feats1 = assign_patches(feats1, w1, h1, x_col="centroid_x", y_col="centroid_y")
    feats2 = assign_patches(feats2, w1, h1, x_col="centroid_x", y_col="centroid_y")

    if use_spatial_clusters:
        feats1, feats2 = assign_clusters_from_round1(
            feats1,
            feats2,
            n_clusters=n_clusters,
            x_col="centroid_x",
            y_col="centroid_y",
            cluster_col="cluster_id",
        )

    match_cfg = MatchingConfig(top_k=top_k, position_weight=position_weight)
    if use_spatial_clusters:
        matches = match_cells_per_cluster(
            feats1, feats2, match_cfg, cluster_col="cluster_id", top_k_per_cluster=top_k_per_patch
        )
    else:
        if top_k_per_patch is not None:
            matches = match_cells_per_patch(feats1, feats2, match_cfg, top_k_per_patch=top_k_per_patch)
        else:
            matches = greedy_match_cells(feats1, feats2, match_cfg)

    if use_topology_filtering:
        if "cell_id" not in feats1.columns:
            feats1 = feats1.copy()
            feats1["cell_id"] = feats1.index
        if "cell_id" not in feats2.columns:
            feats2 = feats2.copy()
            feats2["cell_id"] = feats2.index

        candidate_matches = pd.DataFrame(
            {
                "cell_id_r1": feats1.loc[matches["idx1"], "cell_id"].to_numpy() if not matches.empty else [],
                "cell_id_r2": feats2.loc[matches["idx2"], "cell_id"].to_numpy() if not matches.empty else [],
            }
        )

        trusted_pairs, neighbor_matches = run_topology_matching_df(
            feats1,
            feats2,
            candidate_matches,
            image_width=w1,
            image_height=h1,
            k_pos_nei=k_pos_nei,
            k_neighbor=k_neighbor,
            tau_pos=tau_pos,
            tau_nei=tau_nei,
            tau_map=tau_map,
            id_r1_col="cell_id_r1",
            id_r2_col="cell_id_r2",
            cell_id_col="cell_id",
            x_col="centroid_x",
            y_col="centroid_y",
        )

        combined_pairs = trusted_pairs.copy()
        neighbor_filtered = neighbor_matches
        if not neighbor_matches.empty and "within_threshold" in neighbor_matches.columns:
            neighbor_filtered = neighbor_matches[neighbor_matches["within_threshold"] == True]
        if not neighbor_filtered.empty:
            combined_pairs = pd.concat([combined_pairs, neighbor_filtered], ignore_index=True)

        if combined_pairs.empty:
            matches = pd.DataFrame(columns=["idx1", "idx2", "cell_id_1", "cell_id_2"])
        else:
            combined_pairs = combined_pairs.drop_duplicates(subset=["cell_id_r1", "cell_id_r2"])
            id_to_idx1 = {feats1.loc[i, "cell_id"]: i for i in feats1.index}
            id_to_idx2 = {feats2.loc[i, "cell_id"]: i for i in feats2.index}
            rows = []
            for _, pair in combined_pairs.iterrows():
                cid1 = pair["cell_id_r1"]
                cid2 = pair["cell_id_r2"]
                idx1 = id_to_idx1.get(cid1)
                idx2 = id_to_idx2.get(cid2)
                if idx1 is None or idx2 is None:
                    continue
                row = {"idx1": idx1, "idx2": idx2, "cell_id_1": cid1, "cell_id_2": cid2}
                for extra in ("L_pos", "L_nei", "dist_norm", "within_threshold", "patch_x", "patch_y"):
                    if extra in pair and not pd.isna(pair[extra]):
                        row[extra] = pair[extra]
                rows.append(row)
            matches = pd.DataFrame(rows).reset_index(drop=True)

    if matches.empty:
        raise ValueError("No matches available after matching; cannot estimate transform.")

    if use_ransac_transform:
        transform, _inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1,
            feats2,
            matches,
            max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches["residual_px"] = residuals
    else:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches, use_scale=True)
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches["residual_px"] = residuals

    if (
        residual_prune_quantile is not None
        and 0.0 < residual_prune_quantile < 1.0
        and len(residuals) >= MIN_MATCHES_FOR_REFINEMENT
    ):
        threshold = float(np.quantile(residuals, residual_prune_quantile))
        keep_mask = residuals <= threshold
        kept = int(keep_mask.sum())
        if kept >= MIN_MATCHES_FOR_REFINEMENT and kept < len(matches):
            matches = matches.loc[keep_mask].reset_index(drop=True)
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches, use_scale=True)

    out_fmt = output_format.lstrip(".")
    if mask_only:
        fixed_metric = (masks1 > 0).astype(np.float32)
        moving_metric = (masks2 > 0).astype(np.float32)
        registered = warp_image_to_fixed(
            moving_metric,
            rotation=transform.rotation,
            translation=transform.translation,
            output_shape=fixed_metric.shape[:2],
            order=0,
        )
        residual_pre = np.abs(fixed_metric - moving_metric)
        residual_post = np.abs(fixed_metric - registered)

        metrics_pre = compute_similarity_metrics(fixed_metric, moving_metric)
        metrics_post = compute_similarity_metrics(fixed_metric, registered)

        save_mask(output_dir / f"fixed_mask.{out_fmt}", masks1)
        save_mask(output_dir / f"moving_mask.{out_fmt}", masks2)
        save_float_image(output_dir / f"registered_mask.{out_fmt}", registered)
        save_float_image(output_dir / f"residual_pre_mask.{out_fmt}", residual_pre)
        save_float_image(output_dir / f"residual_post_mask.{out_fmt}", residual_post)
    else:
        fixed_metric = prepare_metric_image(fixed_metric)
        moving_metric = prepare_metric_image(moving_metric)
        registered = warp_image_to_fixed(
            moving_metric,
            rotation=transform.rotation,
            translation=transform.translation,
            output_shape=fixed_metric.shape[:2],
        )

        residual_pre = np.abs(fixed_metric.astype(np.float32) - moving_metric.astype(np.float32))
        residual_post = np.abs(fixed_metric.astype(np.float32) - registered.astype(np.float32))

        metrics_pre = compute_similarity_metrics(fixed_metric, moving_metric)
        metrics_post = compute_similarity_metrics(fixed_metric, registered)

        save_float_image(output_dir / f"registered.{out_fmt}", registered)
        save_float_image(output_dir / f"residual_pre.{out_fmt}", residual_pre)
        save_float_image(output_dir / f"residual_post.{out_fmt}", residual_post)

    metrics = {
        "pre": metrics_pre,
        "post": metrics_post,
        "translation": [float(transform.translation[0]), float(transform.translation[1])],
        "rotation": transform.rotation.tolist(),
        "mask_only": bool(mask_only),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if mask_only:
        print(f"Saved masks to {output_dir / f'fixed_mask.{out_fmt}'} and {output_dir / f'moving_mask.{out_fmt}'}")
        print(
            f"Saved registered mask to {output_dir / f'registered_mask.{out_fmt}'}"
        )
        print(
            f"Saved residuals to {output_dir / f'residual_pre_mask.{out_fmt}'} and "
            f"{output_dir / f'residual_post_mask.{out_fmt}'}"
        )
    else:
        print(f"Saved registered image to {output_dir / f'registered.{out_fmt}'}")
        print(f"Saved residuals to {output_dir / f'residual_pre.{out_fmt}'} and {output_dir / f'residual_post.{out_fmt}'}")
    print(f"Saved metrics to {metrics_path}")
    print(f"SSIM pre/post: {metrics_pre['ssim']:.4f} -> {metrics_post['ssim']:.4f}")
    print(f"NRMSE pre/post: {metrics_pre['nrmse']:.4f} -> {metrics_post['nrmse']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare moving vs fixed images before and after existing registration."
    )
    parser.add_argument("--fixed", type=Path, required=True, help="Path to fixed image.")
    parser.add_argument("--moving", type=Path, required=True, help="Path to moving image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/registration_compare"))
    parser.add_argument("--gpu", action="store_true", help="Use GPU for Cellpose segmentation if available.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-per-patch", type=int, default=TOP_K_PER_PATCH)
    parser.add_argument("--position-weight", type=float, default=DEFAULT_POSITION_WEIGHT)
    parser.add_argument("--use-spatial-clusters", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=9)
    parser.add_argument("--no-topology-filtering", action="store_true", help="Disable topology neighbor mapping.")
    parser.add_argument("--k-pos-nei", type=int, default=5)
    parser.add_argument("--k-neighbor", type=int, default=5)
    parser.add_argument("--tau-pos", type=float, default=0.5)
    parser.add_argument("--tau-nei", type=float, default=0.3)
    parser.add_argument("--tau-map", type=float, default=0.3)
    parser.add_argument("--use-ransac-transform", action="store_true")
    parser.add_argument("--ransac-max-trials", type=int, default=1000)
    parser.add_argument("--ransac-residual-threshold", type=float, default=2.0)
    parser.add_argument(
        "--residual-prune-quantile",
        type=float,
        default=DEFAULT_RESIDUAL_PRUNE_QUANTILE,
        help="Quantile for pruning high-residual matches (0 disables).",
    )
    parser.add_argument("--output-format", type=str, default="tif", help="Image format for outputs.")
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Evaluate and export mask-based comparisons only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_compare(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_dir=args.output_dir,
        top_k=args.top_k,
        top_k_per_patch=args.top_per_patch,
        position_weight=args.position_weight,
        use_spatial_clusters=args.use_spatial_clusters,
        n_clusters=args.n_clusters,
        use_topology_filtering=not args.no_topology_filtering,
        k_pos_nei=args.k_pos_nei,
        k_neighbor=args.k_neighbor,
        tau_pos=args.tau_pos,
        tau_nei=args.tau_nei,
        tau_map=args.tau_map,
        use_ransac_transform=args.use_ransac_transform,
        ransac_max_trials=args.ransac_max_trials,
        ransac_residual_threshold=args.ransac_residual_threshold,
        residual_prune_quantile=args.residual_prune_quantile,
        use_gpu=args.gpu,
        output_format=args.output_format,
        mask_only=args.mask_only,
    )


if __name__ == "__main__":
    main()

