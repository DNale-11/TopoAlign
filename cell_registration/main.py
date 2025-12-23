from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .matching import (
    MatchingConfig,
    greedy_match_cells,
    match_cells_per_patch,
    match_cells_per_cluster,
    assign_patches,
    PATCH_GRID
)
from .registration import RigidTransform, estimate_rigid_transform_from_matches
from .robust_alignment import perform_global_registration, apply_transform_to_coordinates
from .segmentation import CellposeSegmenter
from .io_utils import infer_image_mode, load_image, project_intensity_max

# Allow running as `python cell_registration/main.py` by setting package context.
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "cell_registration"

# Single source of truth for defaults.
DEFAULT_IMG1_PATH = Path("1_ch5.tif")
DEFAULT_IMG2_PATH = Path("2_ch5.tif")
DEFAULT_TOP_K = 50
DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_SAVE_MATCH_TABLE = Path("outputs/top_matches.csv")
DEFAULT_SAVE_MATCH_OVERLAY = Path("outputs/match_overlay")
DEFAULT_SAVE_SEGMENTATION_PREFIX = Path("outputs/segmentation")
DEFAULT_SAVE_MATCH_PLOT = Path("outputs/match_plot")
DEFAULT_SAVE_REGISTRATION_OVERLAY = Path("outputs/registration_overlay")
DEFAULT_SAVE_FEATURES_DIR = Path("outputs")
DEFAULT_RESIDUAL_PRUNE_QUANTILE = 0.9
MIN_MATCHES_FOR_REFINEMENT = 3
TOP_K_PER_PATCH = 6

def assign_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    cols: list[str],
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Assign spatial clusters using KMeans."""
    out = df.copy()
    n_eff = min(n_clusters, len(out))
    if n_eff <= 1:
        out[cluster_col] = 0
        return out

    coords = out[cols].to_numpy()
    kmeans = KMeans(n_clusters=n_eff, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(coords)
    out[cluster_col] = labels
    return out


def assign_clusters_from_round1(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    n_clusters: int,
    cols: list[str],
    cluster_col: str = "cluster_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit KMeans on round1 and assign to round2."""
    feats1_out = feats1.copy()
    feats2_out = feats2.copy()
    n_eff = min(n_clusters, len(feats1_out))
    if n_eff <= 1:
        feats1_out[cluster_col] = 0
        feats2_out[cluster_col] = 0
        return feats1_out, feats2_out

    coords1 = feats1_out[cols].to_numpy()
    kmeans = KMeans(n_clusters=n_eff, random_state=0, n_init="auto")
    kmeans.fit(coords1)
    feats1_out[cluster_col] = kmeans.labels_
    if len(feats2_out) == 0:
        feats2_out[cluster_col] = pd.Series(dtype=int)
    else:
        coords2 = feats2_out[cols].to_numpy()
        feats2_out[cluster_col] = kmeans.predict(coords2)
    return feats1_out, feats2_out


def compute_match_residuals(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
    transform: RigidTransform,
) -> np.ndarray:
    """Compute Euclidean residuals."""
    if matches.empty:
        return np.array([], dtype=float)

    if "centroid_z" in feats1.columns:
        cols = ["centroid_z", "centroid_y", "centroid_x"]
    else:
        cols = ["centroid_y", "centroid_x"]

    pts1 = feats1.iloc[matches["idx1"]][cols].to_numpy(dtype=float)
    pts2 = feats2.iloc[matches["idx2"]][cols].to_numpy(dtype=float)

    # x_new = (R @ x_old.T).T + t
    pts2_warp = (transform.rotation @ pts2.T).T + transform.translation
    return np.linalg.norm(pts1 - pts2_warp, axis=1)


def run_pipeline(
    img1_path: Path = DEFAULT_IMG1_PATH,
    img2_path: Path = DEFAULT_IMG2_PATH,
    top_k: int = DEFAULT_TOP_K,
    top_k_per_patch: int | None = TOP_K_PER_PATCH,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    napari_view: bool = False,
    segmentation_only: bool = False,
    save_match_table: Path = DEFAULT_SAVE_MATCH_TABLE,
    save_match_overlay_path: Path = DEFAULT_SAVE_MATCH_OVERLAY,
    save_segmentation_prefix: Path = DEFAULT_SAVE_SEGMENTATION_PREFIX,
    save_match_plot_path: Path = DEFAULT_SAVE_MATCH_PLOT,
    save_registration_overlay_path: Path = DEFAULT_SAVE_REGISTRATION_OVERLAY,
    save_features_dir: Path | None = DEFAULT_SAVE_FEATURES_DIR,
    residual_prune_quantile: float | None = DEFAULT_RESIDUAL_PRUNE_QUANTILE,
    use_spatial_clusters: bool = False,
    n_clusters: int = 9,

    # 3D specific args
    do_3d: bool = True,
    flow3d_smooth: float = 0.0,
    stitch_threshold: float = 0.0,
    min_size: int = 15,
) -> None:

    # Update config with arguments
    DEFAULT_CELLPOSE_CONFIG.do_3D = do_3d
    DEFAULT_CELLPOSE_CONFIG.flow3D_smooth = flow3d_smooth
    DEFAULT_CELLPOSE_CONFIG.stitch_threshold = stitch_threshold
    DEFAULT_CELLPOSE_CONFIG.min_size = min_size

    segmenter = CellposeSegmenter(DEFAULT_CELLPOSE_CONFIG)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    mode1 = infer_image_mode(img1)
    mode2 = infer_image_mode(img2)

    if mode1 != mode2:
        raise ValueError(
            f"Image modes must match. Image1 is {mode1}, Image2 is {mode2}. "
            "Mixing 2D and 3D images is not supported. Please ensure both are 2D or both are 3D Z-stacks."
        )

    is_3d = (mode1 == "3d_zstack")

    print(f"Processing images. Mode 1: {mode1}, Mode 2: {mode2}. 3D Pipeline: {is_3d}")

    if is_3d:
        # Use segment_zstack which returns 3D masks
        # Note: segment_file calls segment_zstack internally if it detects 3D
        masks1, _, _ = segmenter.segment_zstack(img1)
        masks2, _, _ = segmenter.segment_zstack(img2)

        # Create overlays for visualization (Max Projection)
        overlay_img1 = project_intensity_max(img1)
        overlay_img2 = project_intensity_max(img2)
    else:
        masks1, _, _ = segmenter.segment_array(img1)
        masks2, _, _ = segmenter.segment_array(img2)    
        overlay_img1 = img1
        overlay_img2 = img2

    if segmentation_only:
        # Save projections
        if save_segmentation_prefix is not None:
            from .visualization import save_segmentation_plot

            # For 3D, we project masks for 2D plot
            if is_3d:
                from .segmentation import project_labels_max
                m1_vis = project_labels_max(masks1)
                m2_vis = project_labels_max(masks2)
            else:
                m1_vis, m2_vis = masks1, masks2

            out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img1.tif")
            out2 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img2.tif")
            save_segmentation_plot(overlay_img1, m1_vis, out1, title="Segmentation image1")
            save_segmentation_plot(overlay_img2, m2_vis, out2, title="Segmentation image2")
            print(f"Saved segmentation plots to {out1} and {out2}")
        return

    # Feature extraction (handles 3D automatically)
    feats1 = compute_cell_features(masks1, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
    feats2 = compute_cell_features(masks2, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)

    print(f"Features extracted: {len(feats1)} cells in img1, {len(feats2)} cells in img2")

    # Global Registration
    print("Running robust global alignment...")
    # Select cols
    feat_cols = ("volume", "area", "major_axis_length", "minor_axis_length", "solidity") if is_3d else \
                ("area", "perimeter", "roundness", "eccentricity", "solidity")

    global_transform = perform_global_registration(feats1, feats2, feature_columns=feat_cols)

    if global_transform is not None:
        print("Global alignment successful.")
        print(f"Initial rotation:\n{global_transform.rotation}")
        print(f"Initial translation: {global_transform.translation}")
        feats2_aligned = apply_transform_to_coordinates(feats2, global_transform)
    else:
        print("Global alignment failed or insufficient matches. Proceeding with raw coordinates.")
        feats2_aligned = feats2.copy()

    # Assign patches / clusters
    # Normalized coords are computed during assign_patches
    feats1 = assign_patches(feats1, masks1.shape)
    feats2_aligned = assign_patches(feats2_aligned, masks1.shape)

    if use_spatial_clusters:
        coord_cols = ["centroid_z", "centroid_y", "centroid_x"] if is_3d else ["centroid_y", "centroid_x"]
        feats1, feats2_aligned = assign_clusters_from_round1(
            feats1,
            feats2_aligned,
            n_clusters=n_clusters,
            cols=coord_cols,
            cluster_col="cluster_id",
        )

    if save_features_dir is not None:
        save_features_dir.mkdir(parents=True, exist_ok=True)
        out1 = save_features_dir / "round1_cells.csv"
        out2 = save_features_dir / "round2_cells.csv"
        feats1.to_csv(out1, index=False)
        feats2_aligned.to_csv(out2, index=False)
        print(f"Saved feature tables to {out1} and {out2}")

    # Matching
    match_cfg = MatchingConfig(top_k=top_k, position_weight=position_weight)

    if use_spatial_clusters:
        matches = match_cells_per_cluster(
            feats1, feats2_aligned, match_cfg, cluster_col="cluster_id", top_k_per_cluster=top_k_per_patch
        )
    else:
        if top_k_per_patch is not None:
            matches = match_cells_per_patch(feats1, feats2_aligned, match_cfg, top_k_per_patch=top_k_per_patch)
        else:
            matches = greedy_match_cells(feats1, feats2_aligned, match_cfg)

    print("Top matches:")
    print(matches.head(top_k))

    if matches.empty:
        print("No matches available after matching; aborting registration.")
        return

    # Final Transform Estimation
    transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    residuals = compute_match_residuals(feats1, feats2, matches, transform)
    matches["residual_px"] = residuals

    # Pruning
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
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
            refined_residuals = compute_match_residuals(feats1, feats2, matches, transform)
            matches["residual_px"] = refined_residuals
            print(f"Pruned high-residual matches (>{threshold:.3f}px); kept {kept} matches.")

    print("Estimated rotation matrix:")
    print(transform.rotation)
    print("Estimated translation vector:")
    print(transform.translation)

    if save_match_table is not None:
        from .visualization import export_match_table
        export_match_table(matches, save_match_table)
        print(f"Saved match table to {save_match_table}")

    # Visualization (Projected for now if 3D, or slices)
    # The existing visualization utilities likely expect 2D images.
    # We can pass the projected images and corresponding 2D features for visualization.
    
    # We'll skip complex 3D vis for now or adapt visualization.py if needed.
    # But since the user only asked to "Organize main function" and "Implement 3D segmentation",
    # I'll stick to projecting for the 2D visualization tools.

    if is_3d:
        # Project features to 2D for vis
        # We need to drop Z from centroids
        feats1_2d = feats1.copy()
        feats2_2d = feats2.copy() # feats2 (original, not aligned, as vis applies matches?)
        # Wait, visualization usually takes original features and draws lines.
        # Match overlay needs 2D points.
        feats1_2d["centroid_x"] = feats1["centroid_x"]
        feats1_2d["centroid_y"] = feats1["centroid_y"]
        feats2_2d["centroid_x"] = feats2["centroid_x"]
        feats2_2d["centroid_y"] = feats2["centroid_y"]

        # But 'matches' refers to indices in these DFs.

        if save_match_overlay_path is not None:
            from .visualization import save_match_overlay
            overlay_path = save_match_overlay_path.with_suffix(".tif")
            save_match_overlay(overlay_img1, overlay_img2, feats1_2d, feats2_2d, matches, overlay_path)
            print(f"Saved match overlay image (2D projection) to {overlay_path}")

    else:
        if save_match_overlay_path is not None:
            from .visualization import save_match_overlay
            overlay_path = save_match_overlay_path.with_suffix(".tif")
            save_match_overlay(overlay_img1, overlay_img2, feats1, feats2, matches, overlay_path)
            print(f"Saved match overlay image to {overlay_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell registration demo using Cellpose-SAM.")
    parser.add_argument("image1", type=Path, nargs="?", default=DEFAULT_IMG1_PATH, help="Path to first image (target).")
    parser.add_argument("image2", type=Path, nargs="?", default=DEFAULT_IMG2_PATH, help="Path to second image (source).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of matches to use.")
    parser.add_argument("--position-weight", type=float, default=DEFAULT_POSITION_WEIGHT, help="Weight for spatial proximity.")
    parser.add_argument("--top-per-patch", type=int, default=TOP_K_PER_PATCH, help="Matches per patch.")
    parser.add_argument("--use-spatial-clusters", action="store_true", help="Use adaptive spatial clustering.")
    parser.add_argument("--n-clusters", type=int, default=9, help="Number of spatial clusters.")

    # 3D
    parser.add_argument("--do-3d", action="store_true", default=True, help="Enable 3D segmentation (default True).")
    parser.add_argument("--no-3d", action="store_false", dest="do_3d", help="Disable 3D segmentation.")
    parser.add_argument("--flow3d-smooth", type=float, default=0.0, help="Smoothing for 3D flows.")
    parser.add_argument("--stitch-threshold", type=float, default=0.0, help="Stitch threshold (if not doing full 3D).")
    parser.add_argument("--min-size", type=int, default=15, help="Min size (voxels/pixels).")

    parser.add_argument("--napari", action="store_true", help="Open napari viewers.")
    parser.add_argument("--segmentation-only", action="store_true", help="Run segmentation only.")
    parser.add_argument("--save-match-table", type=Path, default=DEFAULT_SAVE_MATCH_TABLE, help="Path to save CSV.")
    parser.add_argument("--save-match-overlay", type=Path, default=DEFAULT_SAVE_MATCH_OVERLAY, help="Path to save overlay.")
    parser.add_argument("--save-segmentation-prefix", type=Path, default=DEFAULT_SAVE_SEGMENTATION_PREFIX, help="Path prefix for seg plots.")
    parser.add_argument("--save-match-plot", type=Path, default=DEFAULT_SAVE_MATCH_PLOT, help="Path to save match plot.")
    parser.add_argument("--save-registration-overlay", type=Path, default=DEFAULT_SAVE_REGISTRATION_OVERLAY, help="Path to save reg overlay.")
    parser.add_argument("--save-features-dir", type=Path, default=DEFAULT_SAVE_FEATURES_DIR, help="Directory to save features.")

    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        args.image1,
        args.image2,
        top_k=args.top_k,
        top_k_per_patch=args.top_per_patch,
        position_weight=args.position_weight,
        napari_view=args.napari,
        segmentation_only=args.segmentation_only,
        save_match_table=args.save_match_table,
        save_match_overlay_path=args.save_match_overlay,
        save_segmentation_prefix=args.save_segmentation_prefix,
        save_match_plot_path=args.save_match_plot,
        save_registration_overlay_path=args.save_registration_overlay,
        save_features_dir=args.save_features_dir,
        use_spatial_clusters=args.use_spatial_clusters,
        n_clusters=args.n_clusters,
        do_3d=args.do_3d,
        flow3d_smooth=args.flow3d_smooth,
        stitch_threshold=args.stitch_threshold,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
