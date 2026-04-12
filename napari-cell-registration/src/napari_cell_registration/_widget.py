"""
napari widgets for cell registration workflows.
"""

from typing import Annotated
from pathlib import Path
from enum import Enum

import napari
import numpy as np
from napari.types import ImageData, LabelsData
from napari.utils.notifications import show_info

from .core import (
    CellFeaturesConfig,
    CellposeConfig,
    CellposeSegmenter,
    MIN_MATCHES_FOR_REFINEMENT,
    apply_rigid_to_points,
    assign_patches,
    cast_warped_like_original,
    compute_cell_features,
    compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    rigid_transform_to_affine,
    two_stage_match_cells,
)
from .core.point_registration import (
    warp_image_with_transform,
    compute_valid_overlap_mask,
    fit_tps_from_matches,
    warp_image_with_tps,
)


class CellposeModel(Enum):
    """Cellpose model options."""

    CPSAM = "cpsam"
    CYTO = "cyto"
    NUCLEI = "nuclei"
    CYTO2 = "cyto2"
    CYTO3 = "cyto3"


def segment_cells_widget(
    viewer: napari.Viewer,
    images: list[ImageData],
    model: CellposeModel = CellposeModel.CPSAM,
    gpu: bool = False,
    diameter: float = 15,
    flow_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = -2.0,
    cellprob_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = 1.0,
    min_size: int = 5,
    save_masks: bool = False,
    output_dir: str = "./segmentation_output",
):
    """
    Segment cells using Cellpose.

    Supports batch processing of multiple images.
    """
    from tifffile import imwrite

    if not images:
        show_info("Please select at least one image layer.")
        return

    show_info("=== Starting Cell Segmentation ===")
    show_info(f"Model: {model.value} | Total images: {len(images)}")

    config = CellposeConfig(
        gpu=gpu,
        pretrained_model=model.value,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    segmenter = CellposeSegmenter(config)

    layer_names = [layer.name for layer in viewer.layers if hasattr(layer, "data")]

    if save_masks:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Masks will be saved to: {output_path.absolute()}")

    for idx, image in enumerate(images):
        progress = f"[{idx + 1}/{len(images)}]"
        layer_name = f"Image_{idx + 1}"
        img_array = np.asarray(image)
        for name in layer_names:
            layer = viewer.layers[name]
            if hasattr(layer, "data"):
                try:
                    layer_data = np.asarray(layer.data)
                    if layer_data.shape == img_array.shape and np.array_equal(layer_data, img_array):
                        layer_name = name
                        break
                except Exception:
                    continue

        show_info(f"{progress} Processing {layer_name}...")
        mask, flows, styles = segmenter.segment_array(img_array)
        n_cells = len(np.unique(mask)) - 1
        mask_name = f"{layer_name}_mask"
        viewer.add_labels(mask, name=mask_name, opacity=0.5)
        show_info(f"{progress} {mask_name}: Found {n_cells} cells")

        if save_masks:
            mask_filename = output_path / f"{layer_name}_mask.tif"
            imwrite(str(mask_filename), mask.astype(np.uint16))
            show_info(f"{progress} Saved: {mask_filename.name}")

    show_info(f"=== Segmentation Complete! Processed {len(images)} image(s) ===")


def registration_workflow_widget(
    viewer: napari.Viewer,
    image_round1: ImageData,
    image_round2: ImageData,
    mask_round1: LabelsData,
    mask_round2: LabelsData,
    top_k: int = 50,
    max_match_distance_px: int = 100,
    position_weight: float = 1.0,
    residual_prune_quantile: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}] = 0.0,
    use_ransac_transform: bool = False,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    min_area: int = 0,
    max_area: int = 0,
    use_tps: bool = True,
    tps_regularization: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.001}] = 0.001,
    save_results: bool = False,
    output_dir: str = "./registration_output",
):
    """Run the current cell-registration workflow on pre-segmented masks."""
    import pandas as pd
    from tifffile import imwrite

    def _add_match_layers(current_matches: pd.DataFrame, fixed_feats: pd.DataFrame, moving_feats: pd.DataFrame) -> None:
        if current_matches.empty:
            return

        matched_pts1 = fixed_feats.loc[current_matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        matched_pts2 = moving_feats.loc[current_matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(matched_pts1, name="Matched Points Round 1", size=8, face_color="yellow")
        viewer.add_points(matched_pts2, name="Matched Points Round 2", size=8, face_color="orange")

        lines = [[matched_pts1[idx], matched_pts2[idx]] for idx in range(len(current_matches))]
        if not lines:
            return

        viewer.add_shapes(
            lines,
            shape_type="line",
            edge_width=1,
            edge_color="cyan",
            name="Match Lines",
        )
        pixel_distances = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
        show_info(
            "  Match line distances: "
            f"min={pixel_distances.min():.1f}px, "
            f"max={pixel_distances.max():.1f}px, "
            f"mean={pixel_distances.mean():.1f}px"
        )

    show_info("=== Starting Cell Registration Workflow ===")

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Results will be saved to: {output_path.absolute()}")

    mask1 = np.asarray(mask_round1)
    mask2 = np.asarray(mask_round2)
    img1 = np.asarray(image_round1)
    img2 = np.asarray(image_round2)

    show_info("[1/5] Extracting features from Round 1...")
    feat_config = CellFeaturesConfig(
        min_area=min_area if min_area > 0 else None,
        max_area=max_area if max_area > 0 else None,
        topology_neighbor_k=3,
    )
    feats1 = compute_cell_features(mask1, feat_config)
    n_cells1 = len(feats1)
    show_info(f"  Round 1: {n_cells1} cells after filtering")

    show_info("[2/5] Extracting features from Round 2...")
    feats2 = compute_cell_features(mask2, feat_config)
    n_cells2 = len(feats2)
    show_info(f"  Round 2: {n_cells2} cells after filtering")

    if not feats1.empty:
        pts1 = feats1[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts1, name="All Points Round 1", size=4, face_color="cyan", opacity=0.5)

    if not feats2.empty:
        pts2 = feats2[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts2, name="All Points Round 2", size=4, face_color="magenta", opacity=0.5)

    if n_cells1 == 0 or n_cells2 == 0:
        show_info("No cells available after feature extraction.")
        return

    show_info("[3/5] Running morphology-guided matching...")
    max_dist = max(1, int(max_match_distance_px))
    match_result = two_stage_match_cells(
        feats1,
        feats2,
        mask1.shape,
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=position_weight,
        top_k=max(1, int(top_k)),
        distance_threshold=None,
        spatial_window_size=float(max_dist),
        min_cells_for_two_stage=10,
        coarse_top_k=max(24, int(top_k)),
        coarse_distance_threshold=2.0,
        coarse_matching_mode="morphology_guided",
        coarse_allow_scale=False,
        coarse_prefer_affine=False,
        coarse_residual_threshold=max(5.0, float(ransac_residual_threshold) * 2.0),
        coarse_max_trials=min(max(int(ransac_max_trials), 200), 2000),
    )
    matches = match_result.matches.copy()

    if len(match_result.coarse_matches) >= 3:
        tx, ty = match_result.coarse_offset_xy
        if match_result.coarse_transform_accepted:
            show_info(
                "  Coarse translation accepted: "
                f"{match_result.coarse_inlier_count}/{len(match_result.coarse_matches)} inliers, "
                f"median residual={match_result.coarse_median_inlier_residual:.2f}px, "
                f"shift=({float(tx):.1f}, {float(ty):.1f}) px"
            )
        else:
            show_info(
                "  Coarse translation rejected: "
                f"{match_result.coarse_inlier_count}/{len(match_result.coarse_matches)} inliers, "
                f"median residual={match_result.coarse_median_inlier_residual:.2f}px"
            )
    else:
        show_info("  Coarse translation skipped; insufficient confident candidates.")

    if not matches.empty and "distance" in matches.columns:
        show_info(
            "  Match distances: "
            f"min={matches['distance'].min():.2f}, "
            f"max={matches['distance'].max():.2f}, "
            f"mean={matches['distance'].mean():.2f}"
        )

    if matches.empty:
        show_info("No matches available after matching; registration aborted.")
        return

    show_info(f"  Selected matches: {len(matches)}")

    show_info("[4/5] Estimating registration transform...")
    if len(matches) < 3:
        show_info(f"  Only {len(matches)} matches found; need at least 3 to estimate a transform.")
        return

    if use_ransac_transform:
        transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1,
            feats2,
            matches,
            max_trials=int(ransac_max_trials),
            residual_threshold=float(ransac_residual_threshold),
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )
        matches = matches.copy()
        matches["ransac_inlier"] = inlier_mask
        inlier_count = int(inlier_mask.sum())
        show_info(f"  RANSAC support: {inlier_count}/{len(matches)} inliers")
        all_residuals = compute_match_residuals(feats1, feats2, matches, transform)
        if inlier_count >= MIN_MATCHES_FOR_REFINEMENT:
            inlier_residuals = all_residuals[inlier_mask]
            inlier_median = float(np.median(inlier_residuals))
            inlier_mad = float(np.median(np.abs(inlier_residuals - inlier_median)))
            robust_scale = max(1.4826 * inlier_mad, 0.5)
            model_threshold = max(
                float(ransac_residual_threshold) * 2.0,
                inlier_median + 3.0 * robust_scale,
            )
            keep_mask = all_residuals <= model_threshold
            keep_count = int(keep_mask.sum())
            if MIN_MATCHES_FOR_REFINEMENT <= keep_count < len(matches):
                matches = matches.loc[keep_mask].reset_index(drop=True)
                transform, _ = estimate_rigid_transform_from_matches_ransac(
                    feats1,
                    feats2,
                    matches,
                    max_trials=int(ransac_max_trials),
                    residual_threshold=float(ransac_residual_threshold),
                    min_inliers=MIN_MATCHES_FOR_REFINEMENT,
                )
                show_info(
                    "  RANSAC model consistency filter: "
                    f"kept {keep_count}/{len(keep_mask)} matches at <= {model_threshold:.2f}px"
                )
        else:
            show_info("  Too few RANSAC inliers to filter matches safely; keeping all matches")
    else:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

    residuals = compute_match_residuals(feats1, feats2, matches, transform)
    matches = matches.copy()
    matches["residual_px"] = residuals

    if (
        0.0 < float(residual_prune_quantile) < 1.0
        and len(matches) >= MIN_MATCHES_FOR_REFINEMENT
    ):
        threshold = float(np.quantile(residuals, float(residual_prune_quantile)))
        keep_mask = residuals <= threshold
        kept = int(keep_mask.sum())
        if kept >= MIN_MATCHES_FOR_REFINEMENT and kept < len(matches):
            matches = matches.loc[keep_mask].reset_index(drop=True)
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
            matches["residual_px"] = compute_match_residuals(feats1, feats2, matches, transform)
            show_info(
                f"  Residual pruning: kept {kept}/{len(residuals)} matches at <= {threshold:.2f}px"
            )

    affine_transform = rigid_transform_to_affine(transform)
    rotation_deg = float(np.degrees(np.arctan2(transform.rotation[1, 0], transform.rotation[0, 0])))
    show_info(
        "  Final transform: "
        f"rotation={rotation_deg:.2f} deg, "
        f"translation=({float(transform.translation[0]):.1f}, {float(transform.translation[1]):.1f}) px"
    )
    if len(matches) > 0:
        show_info(
            "  Residuals: "
            f"min={matches['residual_px'].min():.2f}px, "
            f"max={matches['residual_px'].max():.2f}px, "
            f"mean={matches['residual_px'].mean():.2f}px"
        )

    show_info("[5/5] Applying transformation and creating overlay...")
    _add_match_layers(matches, feats1, feats2)

    if use_tps and len(matches) >= 3:
        # --- TPS warp: non-rigid, landmark-guided ---
        show_info("  Using TPS warp (non-rigid, landmark-guided)...")
        pts_fixed_xy = feats1.loc[matches["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        pts_moving_xy = feats2.loc[matches["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        tps = fit_tps_from_matches(
            pts_fixed_xy,
            pts_moving_xy,
            output_shape=mask1.shape[:2],
            rigid_transform=affine_transform,
            regularization=float(tps_regularization),
            n_boundary_per_side=4,
            add_boundary_anchors_flag=True,
        )
        img2_warped = warp_image_with_tps(img2, tps, mask1.shape[:2], order=1)
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        mask2_warped = warp_image_with_tps(
            mask2.astype(np.int32), tps, mask1.shape[:2], order=0
        )
    else:
        # --- Fallback: global rigid affine warp ---
        if use_tps:
            show_info("  Too few matches for TPS – falling back to rigid affine warp.")
        img2_warped = warp_image_with_transform(img2, affine_transform, mask1.shape[:2], order=1)
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        mask2_warped = warp_image_with_transform(mask2.astype(np.int32), affine_transform, mask1.shape[:2], order=0)

    mask2_registered = np.rint(mask2_warped).astype(np.int32)

    # Full Fusion: Where the registered moving mask has no data (background 0), retain the fixed mask.
    # This naturally handles both the out-of-bounds boundaries and the spaces between moving cells.
    mask2_registered = np.where(mask2_registered == 0, mask1, mask2_registered)

    # For the image, fuse using maximum intensity projection (keeps signals from both)
    if img1.shape == img2_registered.shape:
        img2_registered = np.maximum(img1, img2_registered)

    image_kwargs = {"name": "Registered Image Round 2", "opacity": 0.5, "blending": "additive"}
    if img2_registered.ndim == 2:
        image_kwargs["colormap"] = "green"
    viewer.add_image(img2_registered, **image_kwargs)
    viewer.add_labels(mask2_registered, name="Registered Mask Round 2", opacity=0.35)

    all_pts_r2_xy = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    all_pts_r2_registered_xy = apply_rigid_to_points(all_pts_r2_xy, transform.rotation, transform.translation)
    viewer.add_points(
        all_pts_r2_registered_xy[:, ::-1],
        name="Registered Points Round 2",
        size=5,
        face_color="red",
        opacity=0.7,
    )

    show_info("  Visualization complete.")

    if save_results:
        show_info("Saving results...")
        img_filename = output_path / "registered_image.tif"
        imwrite(str(img_filename), img2_registered)
        show_info(f"  Saved: {img_filename.name}")

        mask_filename = output_path / "registered_mask.tif"
        imwrite(str(mask_filename), mask2_registered)
        show_info(f"  Saved: {mask_filename.name}")

        feats1_csv = output_path / "features_round1.csv"
        feats1.to_csv(feats1_csv, index=False)
        show_info(f"  Saved: {feats1_csv.name}")

        feats2_csv = output_path / "features_round2.csv"
        feats2.to_csv(feats2_csv, index=False)
        show_info(f"  Saved: {feats2_csv.name}")

        matches_csv = output_path / "matches.csv"
        matches.to_csv(matches_csv, index=False)
        show_info(f"  Saved: {matches_csv.name} ({len(matches)} matches)")

        registered_feats2 = feats2.copy()
        registered_feats2["centroid_x"] = all_pts_r2_registered_xy[:, 0]
        registered_feats2["centroid_y"] = all_pts_r2_registered_xy[:, 1]
        registered_feats2["pos_x_norm"] = registered_feats2["centroid_x"] / float(max(mask1.shape[1], 1))
        registered_feats2["pos_y_norm"] = registered_feats2["centroid_y"] / float(max(mask1.shape[0], 1))
        registered_feats2 = assign_patches(registered_feats2, mask1.shape[1], mask1.shape[0])

        registered_feats_csv = output_path / "registered_features_round2.csv"
        registered_feats2.to_csv(registered_feats_csv, index=False)
        show_info(f"  Saved: {registered_feats_csv.name}")

        pts_r2_reg_df = registered_feats2.loc[:, ["cell_id", "centroid_y", "centroid_x"]]
        reg_pts_csv = output_path / "registered_centroids_round2.csv"
        pts_r2_reg_df.to_csv(reg_pts_csv, index=False)
        show_info(f"  Saved: {reg_pts_csv.name}")

        transform_file = output_path / "transform_matrix.txt"
        with open(transform_file, "w", encoding="utf-8") as handle:
            handle.write("Affine Transform Matrix:\n")
            handle.write(f"{affine_transform.params}\n\n")
            handle.write("Rotation Matrix:\n")
            handle.write(f"{transform.rotation}\n\n")
            handle.write("Translation Vector:\n")
            handle.write(f"{transform.translation}\n")
        show_info(f"  Saved: {transform_file.name}")

    show_info("=== Registration Complete! ===")
    show_info("Check the registered image, mask, and point layers for results.")
