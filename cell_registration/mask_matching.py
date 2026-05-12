"""Mask-based cell matching using IoU and shape descriptors.

This module matches cells between two segmentation rounds by directly
comparing the overlap and shape of their masks, rather than relying on
coarse scalar morphological features (area, perimeter, etc.).

Algorithm:
    1. Spatial candidate selection: for each fixed cell, find moving cells
       whose centroids are within a search radius.
    2. Mask IoU scoring: compute pixel-level intersection-over-union between
       each candidate pair.
    3. Shape descriptor refinement: use Hu moments for additional shape
       similarity scoring.
    4. Greedy 1-to-1 assignment by combined score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from skimage.measure import moments_hu, regionprops_table


@dataclass
class MaskMatchConfig:
    """Configuration for mask-based cell matching."""

    # Maximum centroid distance (pixels) to consider a candidate match.
    search_radius: float = 50.0

    # Minimum IoU to accept a match.
    min_iou: float = 0.1

    # Weight for IoU component in combined score (higher = prefer IoU).
    iou_weight: float = 1.0

    # Weight for Hu moment distance in combined score.
    hu_weight: float = 0.3

    # Weight for area ratio penalty in combined score.
    area_weight: float = 0.2

    # Maximum area ratio deviation to accept (e.g. 3.0 means area can differ by 3x).
    max_area_ratio: float = 3.0

    # Expand bounding boxes by this many pixels when computing IoU.
    bbox_pad: int = 5


@dataclass
class MaskMatchResult:
    """Result of mask-based cell matching."""

    matches: pd.DataFrame  # columns: cell_id_fixed, cell_id_moving, iou, hu_dist, score
    fixed_unmatched: list[int] = field(default_factory=list)
    moving_unmatched: list[int] = field(default_factory=list)
    n_fixed: int = 0
    n_moving: int = 0


def _extract_cell_binary(mask: np.ndarray, label: int) -> np.ndarray:
    """Extract a binary mask for a single cell label."""
    return (mask == label).astype(np.uint8)


def _get_cell_bbox(slices: tuple[slice, slice], pad: int, mask_shape: tuple[int, int]) -> tuple[slice, slice]:
    """Expand a bounding box by pad pixels, clipping to image bounds."""
    h, w = mask_shape
    r_start = max(0, slices[0].start - pad)
    r_stop = min(h, slices[0].stop + pad)
    c_start = max(0, slices[1].start - pad)
    c_stop = min(w, slices[1].stop + pad)
    return (slice(r_start, r_stop), slice(c_start, c_stop))


def compute_mask_iou(
    mask_fixed: np.ndarray,
    mask_moving: np.ndarray,
    label_fixed: int,
    label_moving: int,
    bbox_fixed: tuple[slice, slice] | None = None,
    bbox_moving: tuple[slice, slice] | None = None,
    pad: int = 5,
) -> float:
    """
    Compute IoU between two cells identified by their labels in respective masks.

    Uses bounding box intersection to avoid scanning the full image.
    """
    h, w = mask_fixed.shape[:2]

    if bbox_fixed is None:
        bbox_fixed = _find_label_bbox(mask_fixed, label_fixed)
    if bbox_moving is None:
        bbox_moving = _find_label_bbox(mask_moving, label_moving)

    if bbox_fixed is None or bbox_moving is None:
        return 0.0

    # Expand bboxes
    bf = _get_cell_bbox(bbox_fixed, pad, (h, w))
    bm = _get_cell_bbox(bbox_moving, pad, (h, w))

    # Compute intersection of bounding boxes
    r_start = max(bf[0].start, bm[0].start)
    r_stop = min(bf[0].stop, bm[0].stop)
    c_start = max(bf[1].start, bm[1].start)
    c_stop = min(bf[1].stop, bm[1].stop)

    if r_start >= r_stop or c_start >= c_stop:
        return 0.0

    # Extract patches in the intersection region
    roi = (slice(r_start, r_stop), slice(c_start, c_stop))
    patch_f = (mask_fixed[roi] == label_fixed)
    patch_m = (mask_moving[roi] == label_moving)

    intersection = np.count_nonzero(patch_f & patch_m)
    if intersection == 0:
        return 0.0

    # Union needs full areas (not just intersection region)
    area_f = np.count_nonzero(mask_fixed[bf] == label_fixed)
    area_m = np.count_nonzero(mask_moving[bm] == label_moving)
    union = area_f + area_m - intersection

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def _find_label_bbox(mask: np.ndarray, label: int) -> tuple[slice, slice] | None:
    """Find bounding box for a specific label in the mask."""
    rows, cols = np.where(mask == label)
    if len(rows) == 0:
        return None
    return (slice(int(rows.min()), int(rows.max()) + 1),
            slice(int(cols.min()), int(cols.max()) + 1))


def compute_hu_moments(mask: np.ndarray, label: int) -> np.ndarray:
    """
    Compute Hu moments for a single cell.

    Returns log-transformed Hu moments (7 values), which are
    translation/rotation/scale invariant shape descriptors.
    """
    binary = (mask == label).astype(np.float64)
    hu = moments_hu(binary)
    # Log-transform for better numerical behavior
    # Use sign-preserving log: sign(h) * log10(|h| + eps)
    eps = 1e-30
    hu_log = np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log


def compute_hu_distance(hu1: np.ndarray, hu2: np.ndarray) -> float:
    """Compute L2 distance between two sets of Hu moments."""
    return float(np.linalg.norm(hu1 - hu2))


def _build_label_info(
    mask: np.ndarray,
    features: pd.DataFrame | None = None,
) -> dict[int, dict]:
    """
    Build a lookup of label -> {centroid, area, bbox} from the mask.

    If features DataFrame is provided, uses centroids from there;
    otherwise computes them from the mask directly.
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]  # exclude background

    # Use find_objects for efficient bbox lookup
    all_slices = find_objects(mask)

    info: dict[int, dict] = {}
    if features is not None and not features.empty:
        feat_lookup = features.set_index("cell_id") if "cell_id" in features.columns else features
        for label in labels:
            label_int = int(label)
            bbox_idx = label_int - 1  # find_objects is 0-indexed
            bbox = all_slices[bbox_idx] if bbox_idx < len(all_slices) and all_slices[bbox_idx] is not None else None

            if label_int in feat_lookup.index:
                row = feat_lookup.loc[label_int]
                cx = float(row["centroid_x"])
                cy = float(row["centroid_y"])
                area = float(row["area"]) if "area" in feat_lookup.columns else None
            else:
                # Fallback: compute from mask
                rows, cols = np.where(mask == label)
                if len(rows) == 0:
                    continue
                cy = float(rows.mean())
                cx = float(cols.mean())
                area = float(len(rows))

            if area is None:
                rows, cols = np.where(mask == label)
                area = float(len(rows))

            info[label_int] = {
                "centroid_x": cx,
                "centroid_y": cy,
                "area": area,
                "bbox": bbox,
            }
    else:
        # Compute everything from mask
        props = regionprops_table(mask, properties=["label", "centroid", "area"])
        df_props = pd.DataFrame(props)
        df_props = df_props.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"})
        for _, row in df_props.iterrows():
            label_int = int(row["label"])
            bbox_idx = label_int - 1
            bbox = all_slices[bbox_idx] if bbox_idx < len(all_slices) and all_slices[bbox_idx] is not None else None
            info[label_int] = {
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "area": float(row["area"]),
                "bbox": bbox,
            }

    return info


def match_cells_by_mask_overlap(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_features: pd.DataFrame | None = None,
    moving_features: pd.DataFrame | None = None,
    config: MaskMatchConfig | None = None,
    verbose: bool = True,
) -> MaskMatchResult:
    """
    Match cells between fixed and moving masks using IoU + shape descriptors.

    Parameters
    ----------
    fixed_mask : np.ndarray
        Label mask from the fixed (reference) round.
    moving_mask : np.ndarray
        Label mask from the moving (registered) round.
    fixed_features : pd.DataFrame, optional
        Feature table for fixed cells (must have cell_id, centroid_x, centroid_y).
    moving_features : pd.DataFrame, optional
        Feature table for moving cells.
    config : MaskMatchConfig, optional
        Matching configuration. Uses defaults if not provided.
    verbose : bool
        Print progress information.

    Returns
    -------
    MaskMatchResult
        Matching results including matched pairs and unmatched cells.
    """
    if config is None:
        config = MaskMatchConfig()

    if verbose:
        print("Building label info for fixed mask...")
    fixed_info = _build_label_info(fixed_mask, fixed_features)
    if verbose:
        print(f"  Found {len(fixed_info)} fixed cells")

    if verbose:
        print("Building label info for moving mask...")
    moving_info = _build_label_info(moving_mask, moving_features)
    if verbose:
        print(f"  Found {len(moving_info)} moving cells")

    if not fixed_info or not moving_info:
        return MaskMatchResult(
            matches=pd.DataFrame(columns=[
                "cell_id_fixed", "cell_id_moving", "iou", "hu_dist",
                "area_ratio", "score", "centroid_dist",
            ]),
            n_fixed=len(fixed_info),
            n_moving=len(moving_info),
        )

    # Build KD-tree on moving cell centroids for fast spatial queries
    moving_labels = sorted(moving_info.keys())
    moving_centroids = np.array([
        [moving_info[label]["centroid_x"], moving_info[label]["centroid_y"]]
        for label in moving_labels
    ])
    tree = cKDTree(moving_centroids)

    # Pre-compute Hu moments for all cells
    if verbose:
        print("Computing Hu moments for fixed cells...")
    fixed_hu: dict[int, np.ndarray] = {}
    for label in fixed_info:
        bbox = fixed_info[label]["bbox"]
        if bbox is not None:
            # Compute Hu on the cropped region for efficiency
            patch = (fixed_mask[bbox] == label).astype(np.float64)
            from skimage.measure import moments_central, moments_normalized, moments_hu as _hu
            m = moments_central(patch)
            mn = moments_normalized(m)
            hu = _hu(mn)
        else:
            hu = moments_hu((fixed_mask == label).astype(np.float64))
        eps = 1e-30
        fixed_hu[label] = np.sign(hu) * np.log10(np.abs(hu) + eps)

    if verbose:
        print("Computing Hu moments for moving cells...")
    moving_hu: dict[int, np.ndarray] = {}
    for label in moving_info:
        bbox = moving_info[label]["bbox"]
        if bbox is not None:
            patch = (moving_mask[bbox] == label).astype(np.float64)
            from skimage.measure import moments_central, moments_normalized, moments_hu as _hu
            m = moments_central(patch)
            mn = moments_normalized(m)
            hu = _hu(mn)
        else:
            hu = moments_hu((moving_mask == label).astype(np.float64))
        eps = 1e-30
        moving_hu[label] = np.sign(hu) * np.log10(np.abs(hu) + eps)

    # Score all candidate pairs
    if verbose:
        print(f"Finding candidates within {config.search_radius}px radius...")

    candidates: list[dict] = []
    n_candidates_total = 0

    for fixed_label, finfo in fixed_info.items():
        query = np.array([finfo["centroid_x"], finfo["centroid_y"]])
        # Find all moving cells within search radius
        neighbor_idxs = tree.query_ball_point(query, config.search_radius)

        if not neighbor_idxs:
            continue

        for idx in neighbor_idxs:
            moving_label = moving_labels[idx]
            minfo = moving_info[moving_label]

            # Area ratio filter
            area_ratio = max(finfo["area"], minfo["area"]) / max(min(finfo["area"], minfo["area"]), 1.0)
            if area_ratio > config.max_area_ratio:
                continue

            # Compute IoU
            iou = compute_mask_iou(
                fixed_mask, moving_mask,
                fixed_label, moving_label,
                bbox_fixed=finfo["bbox"],
                bbox_moving=minfo["bbox"],
                pad=config.bbox_pad,
            )

            if iou < config.min_iou:
                continue

            # Compute Hu moment distance
            hu_dist = compute_hu_distance(fixed_hu[fixed_label], moving_hu[moving_label])

            # Centroid distance
            centroid_dist = float(np.linalg.norm(query - moving_centroids[idx]))

            # Combined score (lower = better)
            # IoU is inverted since higher IoU = better match
            score = (
                config.iou_weight * (1.0 - iou)
                + config.hu_weight * hu_dist
                + config.area_weight * (area_ratio - 1.0)
            )

            candidates.append({
                "cell_id_fixed": fixed_label,
                "cell_id_moving": moving_label,
                "iou": iou,
                "hu_dist": hu_dist,
                "area_ratio": area_ratio,
                "score": score,
                "centroid_dist": centroid_dist,
            })
            n_candidates_total += 1

    if verbose:
        print(f"  Total candidate pairs: {n_candidates_total}")

    if not candidates:
        return MaskMatchResult(
            matches=pd.DataFrame(columns=[
                "cell_id_fixed", "cell_id_moving", "iou", "hu_dist",
                "area_ratio", "score", "centroid_dist",
            ]),
            fixed_unmatched=sorted(fixed_info.keys()),
            moving_unmatched=sorted(moving_info.keys()),
            n_fixed=len(fixed_info),
            n_moving=len(moving_info),
        )

    # Greedy 1-to-1 assignment
    if verbose:
        print("Running greedy 1-to-1 assignment...")
    candidates_sorted = sorted(candidates, key=lambda c: c["score"])

    used_fixed: set[int] = set()
    used_moving: set[int] = set()
    matched_rows: list[dict] = []

    for cand in candidates_sorted:
        fid = cand["cell_id_fixed"]
        mid = cand["cell_id_moving"]
        if fid in used_fixed or mid in used_moving:
            continue
        matched_rows.append(cand)
        used_fixed.add(fid)
        used_moving.add(mid)

    match_df = pd.DataFrame(matched_rows)
    fixed_unmatched = sorted(set(fixed_info.keys()) - used_fixed)
    moving_unmatched = sorted(set(moving_info.keys()) - used_moving)

    if verbose:
        print(f"\nMatching summary:")
        print(f"  Fixed cells:     {len(fixed_info)}")
        print(f"  Moving cells:    {len(moving_info)}")
        print(f"  Matched pairs:   {len(matched_rows)}")
        print(f"  Fixed unmatched: {len(fixed_unmatched)}")
        print(f"  Moving unmatched:{len(moving_unmatched)}")
        if not match_df.empty:
            print(f"  IoU mean:        {match_df['iou'].mean():.3f}")
            print(f"  IoU median:      {match_df['iou'].median():.3f}")
            print(f"  IoU > 0.5:       {(match_df['iou'] > 0.5).sum()}")
            print(f"  IoU > 0.3:       {(match_df['iou'] > 0.3).sum()}")

    return MaskMatchResult(
        matches=match_df,
        fixed_unmatched=fixed_unmatched,
        moving_unmatched=moving_unmatched,
        n_fixed=len(fixed_info),
        n_moving=len(moving_info),
    )


def match_masks_pipeline(
    fixed_mask_path: str | Path,
    moving_mask_path: str | Path,
    fixed_features_path: str | Path | None = None,
    moving_features_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    config: MaskMatchConfig | None = None,
    verbose: bool = True,
) -> MaskMatchResult:
    """
    Full pipeline: load masks and features, match cells, optionally save results.

    Parameters
    ----------
    fixed_mask_path : path
        Path to fixed (reference) label mask TIFF.
    moving_mask_path : path
        Path to moving (registered) label mask TIFF.
    fixed_features_path : path, optional
        Path to fixed features CSV. If None, features are computed from mask.
    moving_features_path : path, optional
        Path to moving features CSV. If None, features are computed from mask.
    output_csv : path, optional
        Path to save match results CSV.
    config : MaskMatchConfig, optional
        Matching configuration.
    verbose : bool
        Print progress information.

    Returns
    -------
    MaskMatchResult
    """
    from tifffile import imread

    if verbose:
        print(f"Loading fixed mask: {fixed_mask_path}")
    fixed_mask = imread(str(fixed_mask_path)).astype(np.int32)
    if verbose:
        print(f"  Shape: {fixed_mask.shape}, labels: {len(np.unique(fixed_mask)) - 1}")

    if verbose:
        print(f"Loading moving mask: {moving_mask_path}")
    moving_mask = imread(str(moving_mask_path)).astype(np.int32)
    if verbose:
        print(f"  Shape: {moving_mask.shape}, labels: {len(np.unique(moving_mask)) - 1}")

    fixed_features = None
    if fixed_features_path is not None:
        fixed_features = pd.read_csv(fixed_features_path)
        if verbose:
            print(f"Loaded fixed features: {len(fixed_features)} rows")

    moving_features = None
    if moving_features_path is not None:
        moving_features = pd.read_csv(moving_features_path)
        if verbose:
            print(f"Loaded moving features: {len(moving_features)} rows")

    result = match_cells_by_mask_overlap(
        fixed_mask, moving_mask,
        fixed_features, moving_features,
        config=config,
        verbose=verbose,
    )

    if output_csv is not None and not result.matches.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.matches.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved matches to {output_csv}")

    return result


def visualize_matches(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    matches: pd.DataFrame,
    output_path: str | Path | None = None,
    max_pairs: int = 50,
    figsize: tuple[float, float] = (20, 10),
) -> None:
    """
    Visualize matched cell pairs side by side.

    Shows fixed mask (left) and moving mask (right) with matched cells
    highlighted in the same color and connected by lines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    if matches.empty:
        print("No matches to visualize.")
        return

    display_matches = matches.head(max_pairs)
    n_pairs = len(display_matches)

    # Create colored overlay images
    h, w = fixed_mask.shape[:2]
    fixed_rgb = np.zeros((h, w, 3), dtype=np.float32)
    moving_rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Generate distinct colors
    colors = []
    for i in range(n_pairs):
        hue = float(i) / max(n_pairs, 1)
        colors.append(hsv_to_rgb([hue, 0.8, 0.9]))

    # Paint matched cells
    for i, (_, row) in enumerate(display_matches.iterrows()):
        fid = int(row["cell_id_fixed"])
        mid = int(row["cell_id_moving"])
        color = colors[i]

        f_pixels = fixed_mask == fid
        m_pixels = moving_mask == mid
        for c in range(3):
            fixed_rgb[f_pixels, c] = color[c]
            moving_rgb[m_pixels, c] = color[c]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(fixed_rgb)
    ax1.set_title(f"Fixed mask ({n_pairs} matched cells)")
    ax1.axis("off")

    ax2.imshow(moving_rgb)
    ax2.set_title(f"Moving mask ({n_pairs} matched cells)")
    ax2.axis("off")

    fig.suptitle(
        f"Matched pairs: {len(matches)} | "
        f"Mean IoU: {matches['iou'].mean():.3f} | "
        f"Median IoU: {matches['iou'].median():.3f}",
        fontsize=14,
    )
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_match_pairs_grid(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    matches: pd.DataFrame,
    output_path: str | Path | None = None,
    n_pairs: int = 20,
    sort_by: str = "iou",
    ascending: bool = False,
    patch_size: int = 80,
) -> None:
    """
    Show individual matched cell pairs in a grid for detailed inspection.

    Each cell in the grid shows the fixed cell mask (green) and moving cell
    mask (red) overlaid, with IoU displayed.
    """
    import matplotlib.pyplot as plt

    if matches.empty:
        print("No matches to visualize.")
        return

    sorted_matches = matches.sort_values(sort_by, ascending=ascending)
    display = sorted_matches.head(n_pairs)
    n = len(display)

    cols = min(5, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, (_, row) in enumerate(display.iterrows()):
        ax = axes[i // cols, i % cols]
        fid = int(row["cell_id_fixed"])
        mid = int(row["cell_id_moving"])
        iou_val = row["iou"]

        # Get bounding box around both cells
        f_ys, f_xs = np.where(fixed_mask == fid)
        m_ys, m_xs = np.where(moving_mask == mid)

        if len(f_ys) == 0 or len(m_ys) == 0:
            ax.axis("off")
            continue

        all_ys = np.concatenate([f_ys, m_ys])
        all_xs = np.concatenate([f_xs, m_xs])
        cy, cx = all_ys.mean(), all_xs.mean()

        half = patch_size // 2
        r0 = max(0, int(cy - half))
        r1 = min(fixed_mask.shape[0], int(cy + half))
        c0 = max(0, int(cx - half))
        c1 = min(fixed_mask.shape[1], int(cx + half))

        # Create RGB overlay: green = fixed, red = moving, yellow = overlap
        patch_h = r1 - r0
        patch_w = c1 - c0
        overlay = np.zeros((patch_h, patch_w, 3), dtype=np.float32)

        f_patch = (fixed_mask[r0:r1, c0:c1] == fid)
        m_patch = (moving_mask[r0:r1, c0:c1] == mid)

        overlay[f_patch, 1] = 1.0  # green = fixed
        overlay[m_patch, 0] = 1.0  # red = moving
        # Overlap will appear yellow (green + red)

        ax.imshow(overlay)
        ax.set_title(f"IoU={iou_val:.2f}\nF:{fid} M:{mid}", fontsize=8)
        ax.axis("off")

    # Hide empty axes
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")

    order_label = "best" if not ascending else "worst"
    fig.suptitle(f"Top {n} {order_label} matches by {sort_by}", fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved pair grid to {output_path}")
    else:
        plt.show()
    plt.close(fig)
