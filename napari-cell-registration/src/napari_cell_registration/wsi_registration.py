"""WSI mask-based landmark search and translation registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from tifffile import imread, imwrite

from .core.config import CellFeaturesConfig, MatchingConfig
from .core.features import compute_cell_features
from .core.registration import RigidTransform
from .core.workflow import cast_warped_like_original, compute_match_residuals


@dataclass
class WSIRegistrationResult:
    matches: pd.DataFrame
    transform: RigidTransform
    residuals: np.ndarray
    registered_image: np.ndarray | None
    registered_mask: np.ndarray
    local_translation_grid: np.ndarray | None = None


def find_wsi_landmark_matches(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    top_k: int = 320,
    max_angle_deg: float = 2.0,
    coverage_grid: int = 4,
    neighbor_k: int = 8,
    neighbor_profile_threshold: float = 0.35,
    max_cell_orientation_diff_deg: float = 10.0,
    min_orientation_eccentricity: float = 0.15,
) -> pd.DataFrame:
    """Find one-to-one real-cell landmark pairs for large-translation WSI masks."""
    feature_columns = tuple(
        col for col in MatchingConfig().feature_columns
        if col in feats1.columns and col in feats2.columns
    )
    if not feature_columns:
        return _empty_wsi_match_frame()

    f1, f2 = _robust_standardize_feature_tables(feats1, feats2, feature_columns)
    if len(f1) == 0 or len(f2) == 0:
        return _empty_wsi_match_frame()

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

    candidate_mask, median_disp = select_main_displacement_cluster(displacements)
    if int(candidate_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[candidate_mask]
    idxs2_flat = idxs2_flat[candidate_mask]
    distances_flat = distances_flat[candidate_mask]
    displacements = displacements[candidate_mask]

    parallel_mask = filter_parallel_displacements(displacements, median_disp, max_angle_deg=max_angle_deg)
    if int(parallel_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[parallel_mask]
    idxs2_flat = idxs2_flat[parallel_mask]
    distances_flat = distances_flat[parallel_mask]
    displacements = displacements[parallel_mask]
    length_mask, _ = filter_long_match_lines(displacements)
    if int(length_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[length_mask]
    idxs2_flat = idxs2_flat[length_mask]
    distances_flat = distances_flat[length_mask]
    displacements = displacements[length_mask]
    residuals = np.linalg.norm(displacements - median_disp[None, :], axis=1)

    order = np.lexsort((distances_flat, residuals))
    used1: set[int] = set()
    used2: set[int] = set()
    rows: list[tuple[int, int, float, float, float, float, float, float, float]] = []
    for pos in order:
        i = int(idxs1[pos])
        j = int(idxs2_flat[pos])
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        dx, dy = displacements[pos]
        line_length = float(np.linalg.norm(displacements[pos]))
        rows.append((
            i,
            j,
            float(distances_flat[pos]),
            float(dx),
            float(dy),
            line_length,
            float(residuals[pos]),
            float(feats1.iloc[i]["cell_id"]),
            float(feats2.iloc[j]["cell_id"]),
        ))

    if not rows:
        return _empty_wsi_match_frame()
    candidates = pd.DataFrame(
        rows,
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "match_line_length_px",
            "displacement_residual_px",
            "cell_id_1",
            "cell_id_2",
        ],
    )
    candidates = add_cell_orientation_scores(candidates, feats1, feats2)
    if "orientation_diff_deg" in candidates.columns:
        reliable_orientation = (
            np.minimum(
                candidates["eccentricity_1"].to_numpy(dtype=float),
                candidates["eccentricity_2"].to_numpy(dtype=float),
            )
            >= float(min_orientation_eccentricity)
        )
        orientation_ok = (~reliable_orientation) | (
            candidates["orientation_diff_deg"].to_numpy(dtype=float) <= float(max_cell_orientation_diff_deg)
        )
        candidates = candidates[orientation_ok].copy()
        if len(candidates) < 3:
            return _empty_wsi_match_frame()

    candidates = add_neighbor_profile_scores(
        candidates,
        feats1,
        feats2,
        neighbor_k=neighbor_k,
    )
    candidates = candidates[
        candidates["neighbor_profile_diff"] <= float(neighbor_profile_threshold)
    ].copy()
    if len(candidates) < 3:
        return _empty_wsi_match_frame()
    matches = select_landmarks_with_grid_coverage(candidates, feats1, top_k=top_k, grid=coverage_grid)
    if matches.empty:
        return _empty_wsi_match_frame()
    matches["idx1"] = matches["idx1"].astype(int)
    matches["idx2"] = matches["idx2"].astype(int)
    return matches


def run_wsi_mask_registration(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_image: np.ndarray | None = None,
    moving_image: np.ndarray | None = None,
    top_k: int = 320,
    max_angle_deg: float = 2.0,
    min_area: int = 0,
    max_area: int = 0,
) -> WSIRegistrationResult:
    feature_config = CellFeaturesConfig(
        min_area=min_area if min_area > 0 else None,
        max_area=max_area if max_area > 0 else None,
        topology_neighbor_k=5,
    )
    feats1 = compute_cell_features(np.asarray(fixed_mask), feature_config)
    feats2 = compute_cell_features(np.asarray(moving_mask), feature_config)
    matches = find_wsi_landmark_matches(feats1, feats2, top_k=top_k, max_angle_deg=max_angle_deg)
    if len(matches) < 3:
        raise ValueError(f"WSI landmark search found {len(matches)} pairs; need at least 3.")

    transform = estimate_wsi_translation_from_matches(matches)
    residuals = compute_match_residuals(feats1, feats2, matches, transform)
    matches = matches.copy()
    matches["residual_px"] = residuals

    local_grid = estimate_local_translation_grid(
        matches,
        fixed_mask.shape[:2],
        grid=4,
        fallback_translation=transform.translation,
    )
    registered_image = None
    if moving_image is not None:
        warped = warp_array_with_translation_grid(
            np.asarray(moving_image),
            local_grid,
            output_shape=fixed_mask.shape[:2],
            order=1,
        )
        registered_image = cast_warped_like_original(warped, np.asarray(moving_image).dtype)
    registered_mask = np.rint(
        warp_array_with_translation_grid(
            np.asarray(moving_mask).astype(np.int32, copy=False),
            local_grid,
            output_shape=fixed_mask.shape[:2],
            order=0,
        )
    ).astype(np.int32)
    return WSIRegistrationResult(
        matches=matches,
        transform=transform,
        residuals=residuals,
        registered_image=registered_image,
        registered_mask=registered_mask,
        local_translation_grid=local_grid,
    )


def select_main_displacement_cluster(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    threshold = max(50.0, med_residual + 3.0 * max(1.4826 * mad_residual, 1.0))
    return refined_residuals <= threshold, refined_median


def estimate_wsi_translation_from_matches(matches: pd.DataFrame) -> RigidTransform:
    """Estimate WSI translation robustly from matched real-cell displacement medians."""
    if matches.empty:
        raise ValueError("No WSI landmark matches provided.")
    if not {"dx", "dy"}.issubset(matches.columns):
        raise ValueError("WSI matches must contain dx and dy columns.")
    translation = np.array(
        [
            float(np.median(matches["dx"].to_numpy(dtype=float))),
            float(np.median(matches["dy"].to_numpy(dtype=float))),
        ],
        dtype=float,
    )
    return RigidTransform(rotation=np.eye(2, dtype=float), translation=translation)


def estimate_local_translation_grid(
    matches: pd.DataFrame,
    image_shape: tuple[int, int],
    grid: int = 4,
    fallback_translation: np.ndarray | None = None,
) -> np.ndarray:
    grid = max(1, int(grid))
    if fallback_translation is None:
        fallback_translation = estimate_wsi_translation_from_matches(matches).translation
    local = np.full((grid, grid, 2), np.nan, dtype=float)
    if {"wsi_grid_x", "wsi_grid_y"}.issubset(matches.columns):
        gx = matches["wsi_grid_x"].to_numpy(dtype=int)
        gy = matches["wsi_grid_y"].to_numpy(dtype=int)
    else:
        h, w = image_shape[:2]
        gx = np.clip(np.floor(matches["fixed_x"].to_numpy(dtype=float) * grid / max(w, 1)).astype(int), 0, grid - 1)
        gy = np.clip(np.floor(matches["fixed_y"].to_numpy(dtype=float) * grid / max(h, 1)).astype(int), 0, grid - 1)

    for y in range(grid):
        for x in range(grid):
            in_cell = (gx == x) & (gy == y)
            if np.any(in_cell):
                local[y, x, 0] = float(np.median(matches.loc[in_cell, "dx"].to_numpy(dtype=float)))
                local[y, x, 1] = float(np.median(matches.loc[in_cell, "dy"].to_numpy(dtype=float)))

    missing = ~np.isfinite(local[..., 0])
    if np.any(missing):
        known_yx = np.argwhere(~missing)
        if len(known_yx) == 0:
            local[:, :, 0] = float(fallback_translation[0])
            local[:, :, 1] = float(fallback_translation[1])
        else:
            for y, x in np.argwhere(missing):
                nearest_idx = int(np.argmin(np.sum((known_yx - np.array([y, x])) ** 2, axis=1)))
                nearest_y, nearest_x = known_yx[nearest_idx]
                local[y, x] = local[nearest_y, nearest_x]
    return local


def warp_array_with_translation_grid(
    arr: np.ndarray,
    translation_grid: np.ndarray,
    output_shape: tuple[int, int],
    order: int,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    h, w = int(output_shape[0]), int(output_shape[1])
    grid_h, grid_w = translation_grid.shape[:2]
    y_centers = (np.arange(grid_h, dtype=float) + 0.5) * h / float(grid_h)
    x_centers = (np.arange(grid_w, dtype=float) + 0.5) * w / float(grid_w)
    interp_dx = RegularGridInterpolator((y_centers, x_centers), translation_grid[..., 0], bounds_error=False, fill_value=None)
    interp_dy = RegularGridInterpolator((y_centers, x_centers), translation_grid[..., 1], bounds_error=False, fill_value=None)

    yy, xx = np.meshgrid(np.arange(h, dtype=float), np.arange(w, dtype=float), indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    dx = interp_dx(points).reshape(h, w)
    dy = interp_dy(points).reshape(h, w)
    src_y = yy - dy
    src_x = xx - dx

    if arr.ndim == 2:
        return map_coordinates(arr.astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0)
    if arr.ndim == 3 and arr.shape[-1] <= 8:
        warped = np.zeros((h, w, arr.shape[-1]), dtype=float)
        for c in range(arr.shape[-1]):
            warped[..., c] = map_coordinates(arr[..., c].astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0)
        return warped
    raise ValueError(f"WSI local translation warp supports 2D or YXC arrays, got shape {arr.shape}.")


def select_landmarks_with_grid_coverage(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    top_k: int,
    grid: int = 4,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    grid = max(1, int(grid))
    top_k = max(1, int(top_k))
    if grid < 2:
        return candidates.sort_values(["displacement_residual_px", "distance"], ascending=True).head(top_k).reset_index(drop=True)

    idx1 = candidates["idx1"].to_numpy(dtype=int)
    xs = feats1.iloc[idx1]["centroid_x"].to_numpy(dtype=float)
    ys = feats1.iloc[idx1]["centroid_y"].to_numpy(dtype=float)
    max_x = max(float(feats1["centroid_x"].max()), 1.0)
    max_y = max(float(feats1["centroid_y"].max()), 1.0)
    gx = np.clip(np.floor(xs * grid / max_x).astype(int), 0, grid - 1)
    gy = np.clip(np.floor(ys * grid / max_y).astype(int), 0, grid - 1)
    patch_ids = gy * grid + gx

    ranked = candidates.copy()
    ranked["_patch_id"] = patch_ids
    ranked = ranked.sort_values(["displacement_residual_px", "distance"], ascending=True)

    per_patch_quota = max(1, int(np.ceil(top_k / float(grid * grid))))
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    for patch_id in range(grid * grid):
        patch_rows = ranked[ranked["_patch_id"] == patch_id].head(per_patch_quota)
        for idx in patch_rows.index.to_list():
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    if len(selected_indices) < top_k:
        for idx in ranked.index.to_list():
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            selected_indices.append(idx_int)
            selected_set.add(idx_int)
            if len(selected_indices) >= top_k:
                break

    out = candidates.loc[selected_indices].copy().reset_index(drop=True)
    idx1_out = out["idx1"].to_numpy(dtype=int)
    xs_out = feats1.iloc[idx1_out]["centroid_x"].to_numpy(dtype=float)
    ys_out = feats1.iloc[idx1_out]["centroid_y"].to_numpy(dtype=float)
    out["wsi_grid_x"] = np.clip(np.floor(xs_out * grid / max_x).astype(int), 0, grid - 1)
    out["wsi_grid_y"] = np.clip(np.floor(ys_out * grid / max_y).astype(int), 0, grid - 1)
    return out


def add_neighbor_profile_scores(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    neighbor_k: int = 8,
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        out["neighbor_profile_diff"] = []
        return out

    coords1 = feats1[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    coords2 = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    profiles1 = _neighbor_distance_profiles(coords1, neighbor_k)
    profiles2 = _neighbor_distance_profiles(coords2, neighbor_k)
    vectors1 = _neighbor_vector_profiles(coords1, neighbor_k)
    vectors2 = _neighbor_vector_profiles(coords2, neighbor_k)
    idx1 = out["idx1"].to_numpy(dtype=int)
    idx2 = out["idx2"].to_numpy(dtype=int)
    p1 = profiles1[idx1]
    p2 = profiles2[idx2]
    denom = np.maximum(np.maximum(p1, p2), 1.0)
    distance_diffs = np.median(np.abs(p1 - p2) / denom, axis=1)

    v1 = _normalize_neighbor_vectors(vectors1[idx1])
    v2 = _normalize_neighbor_vectors(vectors2[idx2])
    pair_dists = np.linalg.norm(v1[:, :, None, :] - v2[:, None, :, :], axis=3)
    vector_diffs = 0.5 * (
        np.median(np.min(pair_dists, axis=2), axis=1)
        + np.median(np.min(pair_dists, axis=1), axis=1)
    )
    out["neighbor_distance_diff"] = np.nan_to_num(distance_diffs, nan=np.inf, posinf=np.inf, neginf=np.inf)
    out["neighbor_vector_diff"] = np.nan_to_num(vector_diffs, nan=np.inf, posinf=np.inf, neginf=np.inf)
    out["neighbor_profile_diff"] = np.maximum(out["neighbor_distance_diff"], out["neighbor_vector_diff"])
    return out


def add_cell_orientation_scores(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    required = {"orientation", "eccentricity"}
    if not required.issubset(feats1.columns) or not required.issubset(feats2.columns):
        return out
    idx1 = out["idx1"].to_numpy(dtype=int)
    idx2 = out["idx2"].to_numpy(dtype=int)
    orient1 = feats1.iloc[idx1]["orientation"].to_numpy(dtype=float)
    orient2 = feats2.iloc[idx2]["orientation"].to_numpy(dtype=float)
    out["orientation_diff_deg"] = _orientation_difference_deg(orient1, orient2)
    out["eccentricity_1"] = feats1.iloc[idx1]["eccentricity"].to_numpy(dtype=float)
    out["eccentricity_2"] = feats2.iloc[idx2]["eccentricity"].to_numpy(dtype=float)
    return out


def _orientation_difference_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    diff = np.abs(angle1 - angle2)
    diff = np.mod(diff, np.pi)
    diff = np.minimum(diff, np.pi - diff)
    return np.degrees(diff)


def _neighbor_distance_profiles(coords: np.ndarray, neighbor_k: int) -> np.ndarray:
    n = len(coords)
    k = max(1, int(neighbor_k))
    if n <= 1:
        return np.zeros((n, k), dtype=float)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff + 1)
    profiles = np.asarray(dists[:, 1:], dtype=float)
    if k_eff < k:
        pad = np.repeat(profiles[:, -1:], k - k_eff, axis=1)
        profiles = np.hstack([profiles, pad])
    return profiles


def _neighbor_vector_profiles(coords: np.ndarray, neighbor_k: int) -> np.ndarray:
    n = len(coords)
    k = max(1, int(neighbor_k))
    if n <= 1:
        return np.zeros((n, k, 2), dtype=float)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k_eff + 1)
    neighbor_idxs = np.asarray(idxs[:, 1:], dtype=int)
    vectors = coords[neighbor_idxs] - coords[:, None, :]
    if k_eff < k:
        pad = np.repeat(vectors[:, -1:, :], k - k_eff, axis=1)
        vectors = np.concatenate([vectors, pad], axis=1)
    return vectors.astype(float, copy=False)


def _normalize_neighbor_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=2)
    scale = np.maximum(np.median(lengths, axis=1), 1.0)
    return vectors / scale[:, None, None]


def filter_long_match_lines(displacements: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    lengths = np.linalg.norm(displacements, axis=1)
    if len(lengths) == 0:
        return np.zeros(0, dtype=bool), {"mean": 0.0, "median": 0.0, "threshold": 0.0}
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    mad_len = float(np.median(np.abs(lengths - median_len)))
    robust_sigma = max(1.4826 * mad_len, 1.0)
    threshold = max(mean_len, median_len + 3.0 * robust_sigma)
    return lengths <= threshold, {
        "mean": mean_len,
        "median": median_len,
        "threshold": float(threshold),
    }


def filter_parallel_displacements(
    displacements: np.ndarray,
    reference_disp: np.ndarray,
    max_angle_deg: float,
) -> np.ndarray:
    norms = np.linalg.norm(displacements, axis=1)
    ref_norm = float(np.linalg.norm(reference_disp))
    if ref_norm < 1.0:
        return np.zeros(len(displacements), dtype=bool)
    valid = norms >= 1.0
    cos_values = np.full(len(displacements), -1.0, dtype=float)
    cos_values[valid] = (
        displacements[valid] @ reference_disp
    ) / (norms[valid] * ref_norm + 1e-8)
    cos_values = np.clip(cos_values, -1.0, 1.0)
    angle_diff = np.degrees(np.arccos(cos_values))
    return valid & (angle_diff <= float(max_angle_deg))


def shift_array_xy(arr: np.ndarray, translation_xy: np.ndarray, order: int) -> np.ndarray:
    tx, ty = float(translation_xy[0]), float(translation_xy[1])
    if arr.ndim == 2:
        shift_values = (ty, tx)
    elif arr.ndim == 3 and arr.shape[-1] <= 8:
        shift_values = (ty, tx, 0.0)
    else:
        raise ValueError(f"WSI translation warp supports 2D or YXC arrays, got shape {arr.shape}.")
    return shift(arr, shift=shift_values, order=order, mode="constant", cval=0.0, prefilter=(order > 1))


def cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WSI registration from real mask cell landmarks.")
    parser.add_argument("--fixed-mask", required=True, type=Path)
    parser.add_argument("--moving-mask", required=True, type=Path)
    parser.add_argument("--fixed-image", type=Path, default=None)
    parser.add_argument("--moving-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("wsi_registration_output"))
    parser.add_argument("--top-k", type=int, default=320)
    parser.add_argument("--max-angle-deg", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-area", type=int, default=0)
    args = parser.parse_args()

    fixed_mask = imread(str(args.fixed_mask)).astype(np.int32)
    moving_mask = imread(str(args.moving_mask)).astype(np.int32)
    fixed_image = imread(str(args.fixed_image)) if args.fixed_image is not None else None
    moving_image = imread(str(args.moving_image)) if args.moving_image is not None else None

    result = run_wsi_mask_registration(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        fixed_image=fixed_image,
        moving_image=moving_image,
        top_k=args.top_k,
        max_angle_deg=args.max_angle_deg,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imwrite(str(args.output_dir / "registered_mask.tif"), result.registered_mask)
    if result.registered_image is not None:
        imwrite(str(args.output_dir / "registered_image.tif"), result.registered_image)
    result.matches.to_csv(args.output_dir / "wsi_landmarks.csv", index=False)
    np.savetxt(args.output_dir / "translation_xy.txt", result.transform.translation[None, :], fmt="%.6f")
    print(f"WSI landmarks: {len(result.matches)}")
    print(f"Translation xy: {result.transform.translation.tolist()}")
    print(f"Residual mean/max: {float(result.residuals.mean()):.3f}/{float(result.residuals.max()):.3f}")


def _empty_wsi_match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "match_line_length_px",
            "displacement_residual_px",
            "orientation_diff_deg",
            "eccentricity_1",
            "eccentricity_2",
            "neighbor_distance_diff",
            "neighbor_vector_diff",
            "neighbor_profile_diff",
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


if __name__ == "__main__":
    cli_main()
