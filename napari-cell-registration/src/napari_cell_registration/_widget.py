"""
napari widgets for cell registration workflows.
"""

from typing import Annotated
from pathlib import Path
from enum import Enum

import napari
import numpy as np
import pandas as pd
from napari.types import ImageData, LabelsData
from napari.utils.notifications import show_info

from .core import (
    CellFeaturesConfig,
    CellposeConfig,
    CellposeSegmenter,
    MatchingConfig,
    MIN_MATCHES_FOR_REFINEMENT,
    apply_rigid_to_points,
    assign_patches,
    cast_warped_like_original,
    compute_cell_features,
    compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells,
    rigid_transform_to_affine,
    two_stage_match_cells,
)
from .core.matching import apply_transform_to_features, _filter_candidate_matches_by_hard_constraints
from .core.point_registration import (
    fit_tps_from_matches,
    warp_image_with_transform,
    warp_image_with_tps,
    compute_valid_overlap_mask,
)
from ._qt_init import apply_default_font


class CellposeModel(Enum):
    """Cellpose model options."""

    CPSAM = "cpsam"
    CYTO = "cyto"
    NUCLEI = "nuclei"
    CYTO2 = "cyto2"
    CYTO3 = "cyto3"


class SegmentationMode(Enum):
    """Segmentation execution modes."""

    AUTO = "auto - chunk only if large"
    FULL_IMAGE = "full image - ignore chunk settings"
    CHUNKED = "chunked - use label stitching"


class RegistrationMode(Enum):
    """Registration workflow modes."""

    AUTO = "auto"
    NORMAL = "normal"
    WSI = "wsi"


def segment_cells_widget(
    viewer: napari.Viewer,
    images: list[ImageData],
    model: CellposeModel = CellposeModel.CPSAM,
    gpu: bool = False,
    diameter: float = 15,
    flow_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = -2.0,
    cellprob_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = 1.0,
    min_size: int = 5,
    mode: SegmentationMode = SegmentationMode.AUTO,
    large_image_threshold_mp: Annotated[float, {"min": 1.0, "max": 1000.0, "step": 1.0}] = 64.0,
    chunk_size: Annotated[int, {"min": 256, "max": 8192, "step": 256}] = 2048,
    chunk_overlap: Annotated[int, {"min": 0, "max": 1024, "step": 32}] = 128,
    stitch_labels: bool = True,
    save_masks: bool = False,
    output_dir: str = "./segmentation_output",
):
    """
    Segment cells using Cellpose.

    Supports batch processing of multiple images. Large 2D images can be
    segmented chunk-by-chunk to reduce peak inference memory.
    """
    from tifffile import imwrite

    apply_default_font()

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

    if save_masks:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Masks will be saved to: {output_path.absolute()}")

    for idx, image in enumerate(images):
        progress = f"[{idx + 1}/{len(images)}]"
        layer_name = _find_layer_name(viewer, image, idx)

        use_chunked, megapixels = _should_use_chunked(
            segmenter,
            image,
            mode,
            large_image_threshold_mp,
        )
        show_info(f"{progress} Processing {layer_name} ({megapixels:.1f} MP)...")
        if use_chunked:
            stitch_text = "on" if stitch_labels else "off"
            show_info(
                f"{progress} Chunked segmentation: chunk={chunk_size}px "
                f"overlap={chunk_overlap}px stitch={stitch_text}"
            )
            mask, flows, styles = segmenter.segment_array_chunked(
                image,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                stitch_labels=stitch_labels,
            )
        else:
            show_info(f"{progress} Full-image segmentation; chunk settings are ignored.")
            mask, flows, styles = segmenter.segment_array(np.asarray(image))
        n_cells = int(mask.max())

        # Truncate long layer names for cleaner UI
        MAX_NAME_LEN = 30
        if len(layer_name) > MAX_NAME_LEN:
            short_name = layer_name[:MAX_NAME_LEN - 3] + "..."
            # Rename the source image layer too
            try:
                viewer.layers[layer_name].name = short_name
            except (KeyError, ValueError):
                pass
            layer_name = short_name

        mask_name = f"{layer_name}_mask"
        viewer.add_labels(mask, name=mask_name, opacity=0.5)
        show_info(f"{progress} {mask_name}: Found {n_cells} cells")

        if save_masks:
            mask_filename = output_path / f"{layer_name}_mask.tif"
            imwrite(str(mask_filename), _mask_for_saving(mask))
            show_info(f"{progress} Saved: {mask_filename.name}")

    show_info(f"=== Segmentation Complete! Processed {len(images)} image(s) ===")


def _find_layer_name(viewer: napari.Viewer, image: ImageData, idx: int) -> str:
    """Find the selected image layer name without materializing large arrays."""
    for layer in viewer.layers:
        if hasattr(layer, "data") and layer.data is image:
            return layer.name
    return f"Image_{idx + 1}"


def _should_use_chunked(
    segmenter: CellposeSegmenter,
    image: ImageData,
    mode: SegmentationMode,
    large_image_threshold_mp: float,
) -> tuple[bool, float]:
    height, width = segmenter.spatial_shape(image)
    megapixels = (height * width) / 1_000_000.0
    mode_value = mode.value if isinstance(mode, SegmentationMode) else str(mode)
    if mode_value.startswith("chunked"):
        return True, megapixels
    if mode_value.startswith("full image"):
        return False, megapixels
    return megapixels >= float(large_image_threshold_mp), megapixels


def _mask_for_saving(mask: np.ndarray) -> np.ndarray:
    """Use uint32 when uint16 would truncate labels."""
    if int(mask.max()) > np.iinfo(np.uint16).max:
        return mask.astype(np.uint32, copy=False)
    return mask.astype(np.uint16, copy=False)


def _should_use_wsi_registration(
    image_shape: tuple[int, ...],
    mode: RegistrationMode,
    wsi_threshold_mp: float,
) -> tuple[bool, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    megapixels = (height * width) / 1_000_000.0
    mode_value = mode.value if isinstance(mode, RegistrationMode) else str(mode)
    mode_value = mode_value.strip().lower()
    if mode_value == RegistrationMode.WSI.value:
        return True, megapixels
    if mode_value == RegistrationMode.NORMAL.value:
        return False, megapixels
    return megapixels >= float(wsi_threshold_mp), megapixels


def _run_wsi_registration(
    viewer: napari.Viewer,
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    top_k: int,
    image_megapixels: float,
) -> None:
    show_info(f"[3/5] WSI landmark search from real mask centroids ({image_megapixels:.1f} MP)...")
    matches = _find_wsi_landmark_matches(feats1, feats2, top_k=top_k)
    if matches.empty:
        show_info("  WSI landmark search failed; no registration was applied.")
        return
    if len(matches) < 3:
        show_info(f"  WSI landmark search found only {len(matches)} pairs; need at least 3.")
        return

    show_info(f"  WSI landmarks selected: {len(matches)} real mask cell pairs")
    _add_wsi_landmark_layers(viewer, feats1, feats2, matches)

    transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    residuals = compute_match_residuals(feats1, feats2, matches, transform)
    matches["residual_px"] = residuals
    tx, ty = transform.translation
    show_info(
        "  WSI translation: "
        f"dx={float(tx):.1f}px dy={float(ty):.1f}px, "
        f"residual mean={float(residuals.mean()):.2f}px max={float(residuals.max()):.2f}px"
    )

    show_info("[4/5] Applying WSI translation registration...")
    import time as _time

    _t0 = _time.perf_counter()
    img2_warped = _shift_array_xy(img2, transform.translation, order=1)
    img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
    image_kwargs = {"name": "Registered Image Round 2 (WSI Translation)", "opacity": 0.5, "blending": "additive"}
    if img2_registered.ndim == 2:
        image_kwargs["colormap"] = "green"
    viewer.add_image(img2_registered, **image_kwargs)

    mask2_warped = _shift_array_xy(mask2.astype(np.int32, copy=False), transform.translation, order=0)
    mask2_registered = np.rint(mask2_warped).astype(np.int32)
    viewer.add_labels(mask2_registered, name="Registered Mask Round 2 (WSI Translation)", opacity=0.35)
    show_info(f"[5/5] WSI registration complete in {_time.perf_counter() - _t0:.2f}s")


def _find_wsi_landmark_matches(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    feature_columns = tuple(
        col for col in MatchingConfig().feature_columns
        if col in feats1.columns and col in feats2.columns
    )
    if not feature_columns:
        show_info("  WSI: no shared morphology feature columns available.")
        return _empty_wsi_match_frame()

    f1, f2 = _robust_standardize_feature_tables(feats1, feats2, feature_columns)
    if len(f1) == 0 or len(f2) == 0:
        return _empty_wsi_match_frame()

    from scipy.spatial import cKDTree

    neighbors_per_cell = min(20, len(f2))
    tree = cKDTree(f2)
    distances, idxs2 = tree.query(f1, k=neighbors_per_cell)
    distances = np.asarray(distances, dtype=float)
    idxs2 = np.asarray(idxs2, dtype=int)
    if distances.ndim == 1:
        distances = distances[:, None]
        idxs2 = idxs2[:, None]

    idxs1 = np.repeat(np.arange(len(feats1), dtype=int), neighbors_per_cell)
    idxs2_flat = idxs2.reshape(-1)
    distances_flat = distances.reshape(-1)
    valid = np.isfinite(distances_flat) & (idxs2_flat >= 0)
    idxs1 = idxs1[valid]
    idxs2_flat = idxs2_flat[valid]
    distances_flat = distances_flat[valid]
    if len(idxs1) < 3:
        show_info(f"  WSI: only {len(idxs1)} morphology candidates.")
        return _empty_wsi_match_frame()

    max_candidates = min(len(idxs1), max(200_000, int(top_k) * 500))
    if len(idxs1) > max_candidates:
        keep = np.argpartition(distances_flat, max_candidates - 1)[:max_candidates]
        idxs1 = idxs1[keep]
        idxs2_flat = idxs2_flat[keep]
        distances_flat = distances_flat[keep]

    pts1 = feats1.iloc[idxs1][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[idxs2_flat][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    displacements = pts1 - pts2
    show_info(f"  WSI morphology candidates: {len(displacements)}")

    candidate_mask, median_disp = _select_main_displacement_cluster(displacements)
    if int(candidate_mask.sum()) < 3:
        show_info("  WSI: no stable dominant displacement cluster found.")
        return _empty_wsi_match_frame()

    idxs1 = idxs1[candidate_mask]
    idxs2_flat = idxs2_flat[candidate_mask]
    distances_flat = distances_flat[candidate_mask]
    displacements = displacements[candidate_mask]
    residuals = np.linalg.norm(displacements - median_disp[None, :], axis=1)

    order = np.lexsort((distances_flat, residuals))
    used1: set[int] = set()
    used2: set[int] = set()
    rows: list[tuple[int, int, float, float, float, float, float, float]] = []
    for pos in order:
        i = int(idxs1[pos])
        j = int(idxs2_flat[pos])
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        dx, dy = displacements[pos]
        rows.append((
            i,
            j,
            float(distances_flat[pos]),
            float(dx),
            float(dy),
            float(residuals[pos]),
            float(feats1.iloc[i]["cell_id"]),
            float(feats2.iloc[j]["cell_id"]),
        ))
        if len(rows) >= int(top_k):
            break

    if not rows:
        return _empty_wsi_match_frame()

    matches = pd.DataFrame(
        rows,
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "displacement_residual_px",
            "cell_id_1",
            "cell_id_2",
        ],
    )
    matches["idx1"] = matches["idx1"].astype(int)
    matches["idx2"] = matches["idx2"].astype(int)
    show_info(
        "  WSI dominant displacement: "
        f"dx={float(median_disp[0]):.1f}px dy={float(median_disp[1]):.1f}px, "
        f"kept={len(matches)} one-to-one landmarks"
    )
    return matches


def _empty_wsi_match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "displacement_residual_px",
            "cell_id_1",
            "cell_id_2",
        ]
    )


def _robust_standardize_feature_tables(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    vals1 = feats1.loc[:, feature_columns].to_numpy(dtype=float)
    vals2 = feats2.loc[:, feature_columns].to_numpy(dtype=float)
    combined = np.vstack([vals1, vals2])
    med = np.nanmedian(combined, axis=0)
    mad = np.nanmedian(np.abs(combined - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    vals1 = np.nan_to_num((vals1 - med[None, :]) / scale[None, :])
    vals2 = np.nan_to_num((vals2 - med[None, :]) / scale[None, :])
    return vals1, vals2


def _select_main_displacement_cluster(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bin_size = 512.0
    bins = np.floor(displacements / bin_size).astype(np.int64)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    best_bin_idx = int(np.argmax(counts))
    seed_mask = inverse == best_bin_idx
    if int(seed_mask.sum()) < 3:
        return seed_mask, np.median(displacements[seed_mask], axis=0)

    seed_median = np.median(displacements[seed_mask], axis=0)
    seed_residuals = np.linalg.norm(displacements - seed_median[None, :], axis=1)
    broad_mask = seed_residuals <= bin_size * 1.5
    if int(broad_mask.sum()) < 3:
        return broad_mask, seed_median

    refined_median = np.median(displacements[broad_mask], axis=0)
    refined_residuals = np.linalg.norm(displacements - refined_median[None, :], axis=1)
    inlier_residuals = refined_residuals[broad_mask]
    med_residual = float(np.median(inlier_residuals))
    mad_residual = float(np.median(np.abs(inlier_residuals - med_residual)))
    threshold = max(500.0, med_residual + 3.0 * max(1.4826 * mad_residual, 1.0))
    return refined_residuals <= threshold, refined_median


def _add_wsi_landmark_layers(
    viewer: napari.Viewer,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    pts1 = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    viewer.add_points(pts1, name="WSI Landmark Round 1", size=9, face_color="yellow")
    viewer.add_points(pts2, name="WSI Landmark Round 2", size=9, face_color="orange")
    lines = [[pts1[idx], pts2[idx]] for idx in range(len(matches))]
    if lines:
        viewer.add_shapes(lines, shape_type="line", edge_width=1, edge_color="cyan", name="WSI Landmark Lines")


def _shift_array_xy(arr: np.ndarray, translation_xy: np.ndarray, order: int) -> np.ndarray:
    from scipy.ndimage import shift

    tx, ty = float(translation_xy[0]), float(translation_xy[1])
    if arr.ndim == 2:
        shift_values = (ty, tx)
    elif arr.ndim == 3 and arr.shape[-1] <= 8:
        shift_values = (ty, tx, 0.0)
    else:
        raise ValueError(f"WSI translation warp supports 2D or YXC arrays, got shape {arr.shape}.")
    return shift(arr, shift=shift_values, order=order, mode="constant", cval=0.0, prefilter=(order > 1))


def registration_workflow_widget(
    viewer: napari.Viewer,
    image_round1: ImageData,
    image_round2: ImageData,
    mask_round1: LabelsData,
    mask_round2: LabelsData,
    top_k: int = 320,
    max_match_distance_px: int = 100,
    position_weight: float = 1.0,
    residual_prune_quantile: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}] = 0.0,
    use_ransac_transform: bool = True,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    registration_mode: RegistrationMode = RegistrationMode.AUTO,
    wsi_threshold_mp: Annotated[float, {"min": 1.0, "max": 5000.0, "step": 1.0}] = 64.0,
    min_area: int = 0,
    max_area: int = 0,
    use_gpu: bool = True,
    save_results: bool = False,
    output_dir: str = "./registration_output",
):
    """Run the current cell-registration workflow on pre-segmented masks."""
    from tifffile import imwrite

    apply_default_font()

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
    from .core import gpu_ops as _gpu_ops
    if use_gpu:
        _torch = _gpu_ops._get_torch_cuda()
        if _torch is not None:
            show_info(f"  GPU accelerated: {_torch.cuda.get_device_name(0)}")
        else:
            show_info("  GPU: not available (using CPU)")
    else:
        show_info("  GPU: disabled by user")
        _gpu_ops.tps_predict_gpu = lambda *a, **k: None
        _gpu_ops.pairwise_cdist_gpu = lambda *a, **k: None

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
        topology_neighbor_k=5,
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

    use_wsi, registration_mp = _should_use_wsi_registration(mask1.shape, registration_mode, wsi_threshold_mp)
    if use_wsi:
        _run_wsi_registration(
            viewer,
            img1,
            img2,
            mask1,
            mask2,
            feats1,
            feats2,
            top_k=max(1, int(top_k)),
            image_megapixels=registration_mp,
        )
        return
    show_info(f"  Normal registration mode ({registration_mp:.1f} MP); using existing workflow.")

    show_info("[3/5] Running morphology-guided matching...")
    max_dist = max(1, int(max_match_distance_px))
    MIN_CONSENSUS_FOR_TPS = 50  # retry with wider window if fewer

    for _window_scale in (1.0, 2.0):
        effective_max_dist = max_dist * _window_scale
        if _window_scale > 1.0:
            show_info(
                f"  Retrying with wider window: {effective_max_dist:.0f}px "
                f"(consensus had too few control points)"
            )
        match_result = two_stage_match_cells(
            feats1,
            feats2,
            mask1.shape,
            feature_weight=1.0,
            topology_weight=0.0,
            position_weight=position_weight,
            top_k=max(1, int(top_k)),
            distance_threshold=None,
            spatial_window_size=float(effective_max_dist),
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

        # --- Guided rematch: re-match on transform-aligned data with patch coverage ---
        if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
            show_info("  Guided rematch: re-matching under estimated transform...")
            initial_affine = rigid_transform_to_affine(transform)
            aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
            rematch_window = max(10.0, float(effective_max_dist) * 0.6)
            if _window_scale > 1.0:
                rematch_window = max(rematch_window, 150.0)
            rematch_config = MatchingConfig(
                feature_weight=1.0,
                topology_weight=0.0,
                position_weight=position_weight,
                top_k=max(1, int(top_k)),
                distance_threshold=None,
                spatial_window_size=rematch_window,
            )
            rematch_matches = greedy_match_cells(
                feats1, aligned_feats2, rematch_config, image_shape=mask1.shape,
                coverage_patch_grid=4,
            )
            if len(rematch_matches) >= MIN_MATCHES_FOR_REFINEMENT:
                transform = estimate_rigid_transform_from_matches(
                    feats1, feats2, rematch_matches,
                )
                rematch_residuals = compute_match_residuals(
                    feats1, feats2, rematch_matches, transform,
                )
                rematch_matches = rematch_matches.copy()
                rematch_matches["residual_px"] = rematch_residuals
                show_info(
                    f"  Guided rematch: {len(rematch_matches)} matches, "
                    f"mean residual={float(rematch_residuals.mean()):.2f}px"
                )
                matches = rematch_matches
            else:
                show_info("  Guided rematch: too few matches; keeping original.")

        affine_transform = rigid_transform_to_affine(transform)
        rotation_deg = float(np.degrees(np.arctan2(transform.rotation[1, 0], transform.rotation[0, 0])))
        show_info(
            "  Final rigid baseline: "
            f"rotation={rotation_deg:.2f} deg, "
            f"translation=({float(transform.translation[0]):.1f}, {float(transform.translation[1]):.1f}) px"
        )
        if len(matches) > 0 and "residual_px" in matches.columns:
            show_info(
                "  Rigid residuals: "
                f"min={matches['residual_px'].min():.2f}px, "
                f"max={matches['residual_px'].max():.2f}px, "
                f"mean={matches['residual_px'].mean():.2f}px"
            )

        # --- Orientation filter ---
        n_before_orient = len(matches)
        if n_before_orient >= MIN_MATCHES_FOR_REFINEMENT:
            matches = _filter_candidate_matches_by_hard_constraints(
                matches,
                max_area_ratio=None,
                max_aspect_ratio_ratio=None,
                max_orientation_diff_deg=5.0,
                min_orientation_eccentricity=0.15,
            )
            n_after_orient = len(matches)
            if n_after_orient < n_before_orient:
                show_info(
                    f"  Orientation filter (<=5°): kept {n_after_orient}/{n_before_orient} matches"
                )
                if n_after_orient >= MIN_MATCHES_FOR_REFINEMENT:
                    transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        # --- Local displacement consistency filter (per-patch) ---
        n_before_disp = len(matches)
        if n_before_disp >= MIN_MATCHES_FOR_REFINEMENT:
            pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][
                ["centroid_x", "centroid_y"]
            ].to_numpy(dtype=float)
            pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][
                ["centroid_x", "centroid_y"]
            ].to_numpy(dtype=float)
            disp_vectors = pts_f - pts_m

            h, w = mask1.shape[:2]
            grid = 4
            px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
            py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
            patch_ids = py * grid + px

            keep_mask = np.ones(n_before_disp, dtype=bool)
            angle_threshold_deg = 5.0
            for pid in range(grid * grid):
                in_patch = patch_ids == pid
                n_in = int(in_patch.sum())
                if n_in < 3:
                    continue
                patch_disp = disp_vectors[in_patch]
                patch_indices = np.where(in_patch)[0]
                patch_norms = np.linalg.norm(patch_disp, axis=1)

                cos_thresh = np.cos(np.radians(angle_threshold_deg))
                votes = np.zeros(n_in, dtype=int)
                for a in range(n_in):
                    if patch_norms[a] < 1.0:
                        continue
                    for b in range(n_in):
                        if patch_norms[b] < 1.0:
                            continue
                        cos_ab = np.dot(patch_disp[a], patch_disp[b]) / (
                            patch_norms[a] * patch_norms[b] + 1e-8
                        )
                        if cos_ab >= cos_thresh:
                            votes[a] += 1

                if votes.max() < 2:
                    continue
                consensus_idx = int(np.argmax(votes))
                consensus_disp = patch_disp[consensus_idx]
                consensus_norm = patch_norms[consensus_idx]

                consensus_members = []
                for k in range(n_in):
                    if patch_norms[k] < 1.0:
                        keep_mask[patch_indices[k]] = False
                        continue
                    cos_val = np.dot(patch_disp[k], consensus_disp) / (
                        patch_norms[k] * consensus_norm + 1e-8
                    )
                    if cos_val < cos_thresh:
                        keep_mask[patch_indices[k]] = False
                    else:
                        consensus_members.append(k)

                if len(consensus_members) >= 3:
                    member_lengths = np.array([patch_norms[k] for k in consensus_members])
                    mean_len = float(np.mean(member_lengths))
                    for k in consensus_members:
                        if abs(patch_norms[k] - mean_len) > 5.0:
                            keep_mask[patch_indices[k]] = False

            n_after_disp = int(keep_mask.sum())
            if n_after_disp >= MIN_MATCHES_FOR_REFINEMENT and n_after_disp < n_before_disp:
                matches = matches.loc[keep_mask].reset_index(drop=True)
                show_info(
                    f"  Displacement consensus filter: kept {n_after_disp}/{n_before_disp} matches"
                )
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        # Check if enough control points for TPS; if not, retry with wider window
        if len(matches) >= MIN_CONSENSUS_FOR_TPS:
            break  # enough points, no retry needed
        show_info(
            f"  Only {len(matches)} control points after filtering "
            f"(need {MIN_CONSENSUS_FOR_TPS})"
        )

    # --- Rigid fallback: relaxed matching + dominant direction ---
    if len(matches) < MIN_CONSENSUS_FOR_TPS:
        show_info("  Rigid fallback: relaxed matching at 100px, finding dominant direction...")
        fb_result = two_stage_match_cells(
            feats1, feats2, mask1.shape,
            feature_weight=1.0, topology_weight=0.0, position_weight=position_weight,
            top_k=max(1, int(top_k)), distance_threshold=None,
            spatial_window_size=float(max_dist),
            min_cells_for_two_stage=10, coarse_top_k=max(24, int(top_k)),
            coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False, coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, float(ransac_residual_threshold) * 2.0),
            coarse_max_trials=min(max(int(ransac_max_trials), 200), 2000),
        )
        fb_matches = fb_result.matches.copy()

        # RANSAC
        fb_transform, _ = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, fb_matches,
            max_trials=int(ransac_max_trials), residual_threshold=float(ransac_residual_threshold),
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )

        # Guided rematch
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            fb_affine = rigid_transform_to_affine(fb_transform)
            fb_aligned = apply_transform_to_features(feats2, fb_affine, mask1.shape)
            fb_cfg = MatchingConfig(
                feature_weight=1.0, topology_weight=0.0, position_weight=position_weight,
                top_k=max(1, int(top_k)), distance_threshold=None,
                spatial_window_size=max(10.0, float(max_dist) * 0.6),
            )
            fb_rematch = greedy_match_cells(feats1, fb_aligned, fb_cfg, image_shape=mask1.shape, coverage_patch_grid=4)
            if len(fb_rematch) >= MIN_MATCHES_FOR_REFINEMENT:
                fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_rematch)
                fb_matches = fb_rematch

        # Orientation filter only (no consensus)
        fb_matches = _filter_candidate_matches_by_hard_constraints(
            fb_matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
            max_orientation_diff_deg=5.0, min_orientation_eccentricity=0.15,
        )
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)

        # Find dominant displacement group (angle≤5°, length≤5px)
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            pts_f = feats1.iloc[fb_matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            pts_m = feats2.iloc[fb_matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            disps = pts_f - pts_m
            norms = np.linalg.norm(disps, axis=1)
            angles = np.arctan2(disps[:, 1], disps[:, 0])
            angle_tol = np.radians(5.0)
            length_tol = 5.0
            n_fb = len(fb_matches)
            best_group = []
            for i in range(n_fb):
                group = []
                for j in range(n_fb):
                    a_diff = abs(angles[i] - angles[j])
                    a_diff = min(a_diff, 2 * np.pi - a_diff)
                    if a_diff <= angle_tol and abs(norms[i] - norms[j]) <= length_tol:
                        group.append(j)
                if len(group) > len(best_group):
                    best_group = group
            show_info(f"  Dominant direction group: {len(best_group)}/{n_fb} matches")
            if len(best_group) >= MIN_MATCHES_FOR_REFINEMENT:
                fb_matches = fb_matches.iloc[best_group].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)
                matches = fb_matches

    affine_transform = rigid_transform_to_affine(transform)
    use_tps = len(matches) >= MIN_CONSENSUS_FOR_TPS

    if use_tps:
        # --- Fit TPS (Thin Plate Spline) for non-rigid registration ---
        show_info("  Fitting TPS non-rigid transform from matched landmarks...")
        pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][
            ["centroid_x", "centroid_y"]
        ].to_numpy(dtype=float)
        pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][
            ["centroid_x", "centroid_y"]
        ].to_numpy(dtype=float)
        tps = fit_tps_from_matches(
            pts_fixed_xy,
            pts_moving_xy,
            output_shape=mask1.shape[:2],
            rigid_transform=affine_transform,
            regularization=1e-3,
            n_boundary_per_side=4,
            add_boundary_anchors_flag=True,
        )
        tps_predicted = tps.predict(pts_fixed_xy)
        tps_residuals = np.linalg.norm(
            tps_predicted - pts_moving_xy, axis=1
        )[: len(pts_fixed_xy)]
        show_info(
            f"  TPS fitted with {len(pts_fixed_xy)} control points + boundary anchors, "
            f"control-point residual: mean={float(tps_residuals.mean()):.3f}px"
        )

    show_info("[5/5] Applying TPS transformation and creating overlay...")
    _add_match_layers(matches, feats1, feats2)

    import time as _time

    if use_tps:
        # Warp image with TPS (non-rigid)
        show_info("  Warping image with TPS...")
        _t0 = _time.perf_counter()
        img2_warped = warp_image_with_tps(img2, tps, mask1.shape[:2], order=1)
        _warp_sec = _time.perf_counter() - _t0
        show_info(f"  TPS image warp completed in {_warp_sec:.2f}s")
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        image_kwargs = {"name": "Registered Image Round 2 (TPS)", "opacity": 0.5, "blending": "additive"}
        if img2_registered.ndim == 2:
            image_kwargs["colormap"] = "green"
        viewer.add_image(img2_registered, **image_kwargs)

        mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
    else:
        # Rigid fallback warp
        from scipy.ndimage import map_coordinates
        show_info(f"  Rigid fallback warp: {len(matches)} matches, using affine")
        _t0 = _time.perf_counter()
        H, W = mask1.shape[:2]
        gy, gx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
        queries_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        inv_affine = np.linalg.inv(affine_transform)
        ones = np.ones((len(queries_xy), 1))
        src_xy = (inv_affine @ np.hstack([queries_xy, ones]).T).T[:, :2]
        src_row = src_xy[:, 1].reshape(H, W)
        src_col = src_xy[:, 0].reshape(H, W)

        img2_warped = map_coordinates(img2.astype(float), [src_row, src_col], order=1, mode="constant", cval=0.0)
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        image_kwargs = {"name": "Registered Image Round 2 (Rigid)", "opacity": 0.5, "blending": "additive"}
        if img2_registered.ndim == 2:
            image_kwargs["colormap"] = "green"
        viewer.add_image(img2_registered, **image_kwargs)

        mask2_warped = map_coordinates(mask2.astype(float), [src_row, src_col], order=0, mode="constant", cval=0.0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
        _warp_sec = _time.perf_counter() - _t0
        show_info(f"  Rigid warp completed in {_warp_sec:.2f}s")

    # Full Fusion
    mask2_registered = np.where(mask2_registered == 0, mask1, mask2_registered)

    if img1.shape == img2_registered.shape:
        img2_registered = np.maximum(img1, img2_registered)
    viewer.add_labels(mask2_registered, name="Registered Mask Round 2", opacity=0.35)

    # Warp ALL round2 centroids
    all_pts_r2_xy = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if use_tps:
        all_pts_r2_registered_xy = tps.predict(all_pts_r2_xy)
    else:
        ones_all = np.ones((len(all_pts_r2_xy), 1))
        all_pts_r2_registered_xy = (affine_transform @ np.hstack([all_pts_r2_xy, ones_all]).T).T[:, :2]
    viewer.add_points(
        all_pts_r2_registered_xy[:, ::-1],
        name="Registered Points Round 2",
        size=5,
        face_color="red",
        opacity=0.7,
    )

    # Draw patch grid lines (4x4)
    h, w = mask1.shape[:2]
    grid = 4
    grid_lines = []
    for i in range(1, grid):
        # Vertical lines: x = i * w / grid (in napari coords: col = x, row = y)
        x = i * w / grid
        grid_lines.append(np.array([[0, x], [h, x]]))
        # Horizontal lines: y = i * h / grid
        y = i * h / grid
        grid_lines.append(np.array([[y, 0], [y, w]]))
    viewer.add_shapes(
        grid_lines, shape_type="line", edge_color="yellow",
        edge_width=2, name="Patch Grid (4x4)", opacity=0.6,
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

        transform_file = output_path / "transform_info.txt"
        with open(transform_file, "w", encoding="utf-8") as handle:
            handle.write("Registration Method: TPS (Thin Plate Spline)\n")
            handle.write(f"Control Points: {len(pts_fixed_xy)}\n")
            handle.write(f"TPS Regularization: 1e-3\n\n")
            handle.write("Rigid Baseline (Affine Transform Matrix):\n")
            handle.write(f"{affine_transform.params}\n\n")
            handle.write("Rotation Matrix:\n")
            handle.write(f"{transform.rotation}\n\n")
            handle.write("Translation Vector:\n")
            handle.write(f"{transform.translation}\n")
        show_info(f"  Saved: {transform_file.name}")

    show_info("=== Registration Complete! ===")
    show_info("Check the registered image, mask, and point layers for results.")
