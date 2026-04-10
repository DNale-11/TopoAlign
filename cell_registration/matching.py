"""Cell matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skimage.transform import AffineTransform

from .point_registration import estimate_robust_transform


@dataclass
class MatchingConfig:
    """Configuration for greedy cell matching based on feature similarity."""

    feature_columns: Tuple[str, ...] = (
        "area",
        "perimeter",
        "roundness",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "aspect_ratio",
        "elongation",
        "equivalent_diameter",
    )
    feature_weight: float = 1.0
    topology_feature_columns: Tuple[str, ...] = (
        "nn_dist_1",
        "nn_dist_2",
        "nn_dist_3",
        "local_density",
        "neighbor_area_ratio_mean",
        "neighbor_roundness_mean",
    )
    topology_weight: float = 0.0
    # Uses normalized positions (pos_x_norm/pos_y_norm).
    position_weight: float = 1.0
    top_k: int = 10
    distance_threshold: float | None = None
    spatial_window_size: float | None = None


@dataclass
class TwoStageMatchResult:
    matches: pd.DataFrame
    coarse_matches: pd.DataFrame
    aligned_df2: pd.DataFrame
    coarse_offset_xy: np.ndarray
    coarse_transform: AffineTransform
    coarse_transform_method: str
    coarse_inlier_count: int
    coarse_inlier_ratio: float
    coarse_median_inlier_residual: float
    coarse_transform_accepted: bool


def _standardize_features(
    df1: pd.DataFrame, df2: pd.DataFrame, feature_columns: Tuple[str, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_columns) == 0:
        return np.zeros((len(df1), 0), dtype=float), np.zeros((len(df2), 0), dtype=float)
    scaler = StandardScaler()
    combined = pd.concat(
        [df1.loc[:, feature_columns], df2.loc[:, feature_columns]], axis=0, ignore_index=True
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]
    return f1, f2


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if len(cols) == 0:
        return
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns {missing}. "
            "Compute features first via `compute_cell_features` (regionprops-based)."
        )


def _pairwise_feature_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: Tuple[str, ...],
) -> np.ndarray:
    if len(feature_columns) == 0:
        return np.zeros((len(df1), len(df2)), dtype=float)
    _ensure_columns(df1, feature_columns)
    _ensure_columns(df2, feature_columns)
    f1, f2 = _standardize_features(df1, df2, feature_columns)
    if f1.shape[1] == 0:
        return np.zeros((len(df1), len(df2)), dtype=float)
    return np.linalg.norm(f1[:, None, :] - f2[None, :, :], axis=2)


def _all_feature_columns(config: MatchingConfig) -> tuple[str, ...]:
    shape_cols = tuple(getattr(config, "feature_columns", ()))
    topology_cols = tuple(getattr(config, "topology_feature_columns", ()))
    return tuple(dict.fromkeys(shape_cols + topology_cols))


def compute_match_distance_matrix(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
) -> np.ndarray:
    shape_cols = tuple(getattr(config, "feature_columns", ()))
    topology_cols = tuple(getattr(config, "topology_feature_columns", ()))
    feature_weight = float(getattr(config, "feature_weight", 1.0) or 0.0)
    topology_weight = float(getattr(config, "topology_weight", 0.0) or 0.0)
    position_weight = float(getattr(config, "position_weight", 0.0) or 0.0)

    dist_matrix = np.zeros((len(df1), len(df2)), dtype=float)
    if feature_weight > 0 and len(shape_cols) > 0:
        dist_matrix += feature_weight * _pairwise_feature_distance(df1, df2, shape_cols)
    if topology_weight > 0 and len(topology_cols) > 0:
        dist_matrix += topology_weight * _pairwise_feature_distance(df1, df2, topology_cols)

    if position_weight > 0:
        _ensure_columns(df1, ("pos_x_norm", "pos_y_norm"))
        _ensure_columns(df2, ("pos_x_norm", "pos_y_norm"))
        coords1 = df1[["pos_x_norm", "pos_y_norm"]].to_numpy()
        coords2 = df2[["pos_x_norm", "pos_y_norm"]].to_numpy()
        pos_dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
        dist_matrix += position_weight * pos_dist

    if config.spatial_window_size is not None:
        _ensure_columns(df1, ("centroid_x", "centroid_y"))
        _ensure_columns(df2, ("centroid_x", "centroid_y"))
        coords1_pixels = df1[["centroid_x", "centroid_y"]].to_numpy()
        coords2_pixels = df2[["centroid_x", "centroid_y"]].to_numpy()
        spatial_dist_pixels = np.linalg.norm(
            coords1_pixels[:, None, :] - coords2_pixels[None, :, :], axis=2
        )
        dist_matrix[spatial_dist_pixels > config.spatial_window_size] = np.inf

    return dist_matrix


def _update_feature_positions(
    df: pd.DataFrame,
    coords_xy: np.ndarray,
    image_shape: Sequence[int],
) -> pd.DataFrame:
    h, w = image_shape[:2]
    out = df.copy()
    out["centroid_x"] = coords_xy[:, 0]
    out["centroid_y"] = coords_xy[:, 1]
    out["pos_x_norm"] = coords_xy[:, 0] / float(max(w, 1))
    out["pos_y_norm"] = coords_xy[:, 1] / float(max(h, 1))
    return out


def _add_patch_coordinates(
    df: pd.DataFrame,
    image_shape: Sequence[int],
    patch_rows: int,
    patch_cols: int,
) -> pd.DataFrame:
    out = df.copy()
    h, w = image_shape[:2]
    x = out["centroid_x"].to_numpy(dtype=float)
    y = out["centroid_y"].to_numpy(dtype=float)
    out["patch_x"] = np.clip(
        np.floor(x * float(max(patch_cols, 1)) / float(max(w, 1))).astype(int),
        0,
        max(patch_cols - 1, 0),
    )
    out["patch_y"] = np.clip(
        np.floor(y * float(max(patch_rows, 1)) / float(max(h, 1))).astype(int),
        0,
        max(patch_rows - 1, 0),
    )
    return out


def _resolve_patch_top_k(
    global_top_k: int,
    patch_rows: int,
    patch_cols: int,
    patch_top_k_per_patch: int | None,
) -> int:
    if patch_top_k_per_patch is not None and patch_top_k_per_patch > 0:
        return int(patch_top_k_per_patch)
    patches = max(int(patch_rows) * int(patch_cols), 1)
    return max(1, int(np.ceil(float(global_top_k) / float(patches))))


def apply_transform_to_features(
    df: pd.DataFrame,
    transform: AffineTransform,
    image_shape: Sequence[int],
) -> pd.DataFrame:
    _ensure_columns(df, ("centroid_x", "centroid_y"))
    coords_xy = df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    coords_xy = transform(coords_xy)
    return _update_feature_positions(df, coords_xy, image_shape)


def estimate_global_offset_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    coarse_matches: pd.DataFrame,
    max_pairs: int = 20,
    mad_factor: float = 3.5,
) -> np.ndarray:
    if coarse_matches.empty:
        return np.zeros(2, dtype=float)

    best = coarse_matches.nsmallest(min(max_pairs, len(coarse_matches)), "distance")
    pts1 = df1.loc[best["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = df2.loc[best["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    offsets = pts1 - pts2
    median = np.median(offsets, axis=0)

    mad = np.median(np.abs(offsets - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    inliers = np.all(np.abs(offsets - median) <= mad_factor * scale, axis=1)
    if int(inliers.sum()) >= 3:
        median = np.median(offsets[inliers], axis=0)

    return median.astype(float, copy=False)


def _estimate_coarse_transform_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    coarse_matches: pd.DataFrame,
    *,
    allow_scale: bool,
    prefer_affine: bool,
    residual_threshold: float,
    similarity_residual_threshold: float | None,
    max_trials: int,
) -> tuple[AffineTransform, np.ndarray, str, int, float]:
    if len(coarse_matches) < 3:
        return AffineTransform(), np.zeros(2, dtype=float), "identity", 0, float("inf")

    pts_fixed = df1.loc[coarse_matches["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving = df2.loc[coarse_matches["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    try:
        robust = estimate_robust_transform(
            pts_fixed,
            pts_moving,
            allow_scale=allow_scale,
            prefer_affine=prefer_affine,
            residual_threshold=residual_threshold,
            similarity_residual_threshold=similarity_residual_threshold,
            max_trials=max_trials,
        )
        transform = robust.transform
        method = robust.method
        inlier_count = robust.inlier_count
        median_inlier_residual = robust.median_inlier_residual
    except Exception:
        offset_xy = estimate_global_offset_from_matches(df1, df2, coarse_matches)
        transform = AffineTransform(translation=(float(offset_xy[0]), float(offset_xy[1])))
        method = "translation"
        inlier_count = 0
        median_inlier_residual = float("inf")

    offset_xy = np.asarray(transform.translation, dtype=float)
    return transform, offset_xy, method, inlier_count, float(median_inlier_residual)


def _should_accept_coarse_transform(
    n_coarse_matches: int,
    inlier_count: int,
    median_inlier_residual: float,
    *,
    min_inlier_count: int,
    min_inlier_ratio: float,
    max_median_inlier_residual: float | None,
) -> bool:
    if n_coarse_matches <= 0:
        return False
    if inlier_count < int(min_inlier_count):
        return False
    if (float(inlier_count) / float(n_coarse_matches)) < float(min_inlier_ratio):
        return False
    if max_median_inlier_residual is not None:
        if not np.isfinite(median_inlier_residual):
            return False
        if float(median_inlier_residual) > float(max_median_inlier_residual):
            return False
    return True


def two_stage_match_cells(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    image_shape: Sequence[int],
    *,
    feature_weight: float = 1.0,
    topology_weight: float = 0.0,
    position_weight: float,
    top_k: int,
    distance_threshold: float | None,
    spatial_window_size: float | None,
    min_cells_for_two_stage: int = 20,
    coarse_top_k: int = 50,
    coarse_distance_threshold: float | None = 2.0,
    coarse_matching_mode: str = "patch",
    coarse_patch_rows: int = 3,
    coarse_patch_cols: int = 3,
    coarse_patch_top_k_per_patch: int | None = None,
    coarse_allow_scale: bool = False,
    coarse_prefer_affine: bool = False,
    coarse_residual_threshold: float = 5.0,
    coarse_similarity_residual_threshold: float | None = None,
    coarse_max_trials: int = 200,
    coarse_min_inlier_count: int = 6,
    coarse_min_inlier_ratio: float = 0.35,
    coarse_max_median_inlier_residual: float | None = None,
) -> TwoStageMatchResult:
    coarse_matches = pd.DataFrame(columns=["idx1", "idx2", "distance"])
    aligned_df2 = df2
    coarse_offset_xy = np.zeros(2, dtype=float)
    coarse_transform = AffineTransform()
    coarse_transform_method = "identity"
    coarse_inlier_count = 0
    coarse_inlier_ratio = 0.0
    coarse_median_inlier_residual = float("inf")
    coarse_transform_accepted = False

    if len(df1) >= min_cells_for_two_stage and len(df2) >= min_cells_for_two_stage:
        coarse_config = MatchingConfig(
            feature_weight=feature_weight,
            topology_weight=topology_weight,
            position_weight=0.0,
            top_k=min(coarse_top_k, len(df1), len(df2)),
            distance_threshold=coarse_distance_threshold,
            spatial_window_size=None if spatial_window_size is None else spatial_window_size * 2.0,
        )
        coarse_strategy = str(coarse_matching_mode).strip().lower()
        if coarse_strategy == "patch":
            patch_rows = max(1, int(coarse_patch_rows))
            patch_cols = max(1, int(coarse_patch_cols))
            fixed_patched = _add_patch_coordinates(df1, image_shape, patch_rows, patch_cols)
            moving_patched = _add_patch_coordinates(df2, image_shape, patch_rows, patch_cols)
            local_top_k = _resolve_patch_top_k(
                coarse_config.top_k,
                patch_rows,
                patch_cols,
                coarse_patch_top_k_per_patch,
            )
            coarse_matches = match_cells_per_patch(
                fixed_patched,
                moving_patched,
                coarse_config,
                top_k_per_patch=local_top_k,
            )
            if len(coarse_matches) > coarse_config.top_k:
                coarse_matches = (
                    coarse_matches.sort_values("distance", ascending=True)
                    .head(coarse_config.top_k)
                    .reset_index(drop=True)
                )
        else:
            coarse_matches = greedy_match_cells(df1, df2, coarse_config)

        if len(coarse_matches) >= 3:
            coarse_transform, coarse_offset_xy, coarse_transform_method, coarse_inlier_count, coarse_median_inlier_residual = (
                _estimate_coarse_transform_from_matches(
                    df1,
                    df2,
                    coarse_matches,
                    allow_scale=coarse_allow_scale,
                    prefer_affine=coarse_prefer_affine,
                    residual_threshold=coarse_residual_threshold,
                    similarity_residual_threshold=coarse_similarity_residual_threshold,
                    max_trials=coarse_max_trials,
                )
            )
            coarse_inlier_ratio = float(coarse_inlier_count) / float(max(len(coarse_matches), 1))
            coarse_transform_accepted = _should_accept_coarse_transform(
                len(coarse_matches),
                coarse_inlier_count,
                coarse_median_inlier_residual,
                min_inlier_count=coarse_min_inlier_count,
                min_inlier_ratio=coarse_min_inlier_ratio,
                max_median_inlier_residual=(
                    coarse_residual_threshold
                    if coarse_max_median_inlier_residual is None
                    else coarse_max_median_inlier_residual
                ),
            )
            if coarse_transform_accepted:
                aligned_df2 = apply_transform_to_features(df2, coarse_transform, image_shape)

    fine_config = MatchingConfig(
        feature_weight=feature_weight,
        topology_weight=topology_weight,
        position_weight=position_weight,
        top_k=top_k,
        distance_threshold=distance_threshold,
        spatial_window_size=spatial_window_size,
    )
    matches = greedy_match_cells(df1, aligned_df2, fine_config)

    return TwoStageMatchResult(
        matches=matches,
        coarse_matches=coarse_matches,
        aligned_df2=aligned_df2,
        coarse_offset_xy=coarse_offset_xy,
        coarse_transform=coarse_transform,
        coarse_transform_method=coarse_transform_method,
        coarse_inlier_count=coarse_inlier_count,
        coarse_inlier_ratio=coarse_inlier_ratio,
        coarse_median_inlier_residual=coarse_median_inlier_residual,
        coarse_transform_accepted=coarse_transform_accepted,
    )


def greedy_match_cells(df1: pd.DataFrame, df2: pd.DataFrame, config: MatchingConfig) -> pd.DataFrame:
    """
    Greedy 1-to-1 matching between two sets of cells using feature similarity.
    """
    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)
    dist_matrix = compute_match_distance_matrix(df1, df2, config)

    pairs = []
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            pairs.append((i, j, dist_matrix[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    used_1 = set()
    used_2 = set()
    matches = []

    for i, j, d in pairs_sorted:
        if i in used_1 or j in used_2:
            continue
        if not np.isfinite(d):
            break
        if config.distance_threshold is not None and d > config.distance_threshold:
            break
        matches.append((i, j, d))
        used_1.add(i)
        used_2.add(j)
        if len(matches) >= config.top_k:
            break

    match_df = pd.DataFrame(matches, columns=["idx1", "idx2", "distance"])
    if match_df.empty:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])
    match_df["cell_id_1"] = df1.iloc[match_df["idx1"]]["cell_id"].to_numpy()
    match_df["cell_id_2"] = df2.iloc[match_df["idx2"]]["cell_id"].to_numpy()

    for col in report_cols:
        match_df[f"{col}_1"] = df1.iloc[match_df["idx1"]][col].to_numpy()
        match_df[f"{col}_2"] = df2.iloc[match_df["idx2"]][col].to_numpy()

    return match_df


def match_cells_per_patch(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    top_k_per_patch: int = 6,
) -> pd.DataFrame:
    """
    Match cells within each patch separately and keep the best N pairs per patch.
    """
    if top_k_per_patch < 1:
        raise ValueError("top_k_per_patch must be >= 1.")

    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)
    _ensure_columns(df1, ("patch_x", "patch_y"))
    _ensure_columns(df2, ("patch_x", "patch_y"))

    rows: list[tuple[int, int, float, int, int]] = []
    patches = sorted(set(zip(df1["patch_x"], df1["patch_y"])) & set(zip(df2["patch_x"], df2["patch_y"])))

    for px, py in patches:
        idxs1 = df1.index[(df1["patch_x"] == px) & (df1["patch_y"] == py)].to_list()
        idxs2 = df2.index[(df2["patch_x"] == px) & (df2["patch_y"] == py)].to_list()
        if not idxs1 or not idxs2:
            continue

        local_df1 = df1.loc[idxs1].reset_index(drop=True)
        local_df2 = df2.loc[idxs2].reset_index(drop=True)
        dist_matrix = compute_match_distance_matrix(local_df1, local_df2, config)

        candidates = []
        for i_local, idx1 in enumerate(idxs1):
            for j_local, idx2 in enumerate(idxs2):
                candidates.append((idx1, idx2, dist_matrix[i_local, j_local]))

        candidates_sorted = sorted(candidates, key=lambda x: x[2])
        used1: set[int] = set()
        used2: set[int] = set()
        kept = 0
        for idx1, idx2, d in candidates_sorted:
            if idx1 in used1 or idx2 in used2:
                continue
            if not np.isfinite(d):
                break
            if config.distance_threshold is not None and d > config.distance_threshold:
                break
            rows.append((idx1, idx2, d, px, py))
            used1.add(idx1)
            used2.add(idx2)
            kept += 1
            if kept >= top_k_per_patch:
                break

    if not rows:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2", "patch_x", "patch_y"])

    match_df = pd.DataFrame(rows, columns=["idx1", "idx2", "distance", "patch_x", "patch_y"])
    match_df["cell_id_1"] = df1.loc[match_df["idx1"], "cell_id"].to_numpy()
    match_df["cell_id_2"] = df2.loc[match_df["idx2"], "cell_id"].to_numpy()

    for col in report_cols:
        match_df[f"{col}_1"] = df1.loc[match_df["idx1"], col].to_numpy()
        match_df[f"{col}_2"] = df2.loc[match_df["idx2"], col].to_numpy()

    return match_df.reset_index(drop=True)


def match_cells_per_cluster(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    cluster_col: str = "cluster_id",
    top_k_per_cluster: int | None = None,
) -> pd.DataFrame:
    """
    Match cells within each spatial cluster separately and keep the best pairs per cluster.
    """
    if cluster_col not in df1.columns or cluster_col not in df2.columns:
        raise ValueError(f"Missing '{cluster_col}' column required for cluster-based matching.")

    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)

    rows: list[dict] = []
    clusters = sorted(set(df1[cluster_col]) & set(df2[cluster_col]))
    for cluster in clusters:
        idxs1 = df1.index[df1[cluster_col] == cluster].to_list()
        idxs2 = df2.index[df2[cluster_col] == cluster].to_list()
        if not idxs1 or not idxs2:
            continue

        local_df1 = df1.loc[idxs1].reset_index(drop=True)
        local_df2 = df2.loc[idxs2].reset_index(drop=True)
        local_matches = greedy_match_cells(local_df1, local_df2, config)
        if local_matches.empty:
            continue

        for _, match in local_matches.iterrows():
            global_idx1 = idxs1[int(match["idx1"])]
            global_idx2 = idxs2[int(match["idx2"])]
            rows.append(
                {
                    "idx1": global_idx1,
                    "idx2": global_idx2,
                    "distance": match["distance"],
                    cluster_col: cluster,
                    "cell_id_1": df1.loc[global_idx1, "cell_id"],
                    "cell_id_2": df2.loc[global_idx2, "cell_id"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2", cluster_col])

    match_df = pd.DataFrame(rows)
    if top_k_per_cluster is not None and top_k_per_cluster > 0:
        kept_groups: list[pd.DataFrame] = []
        for _, group in match_df.groupby(cluster_col):
            kept_groups.append(group.sort_values("distance", ascending=True).head(top_k_per_cluster))
        match_df = pd.concat(kept_groups, ignore_index=True)

    for col in report_cols:
        match_df[f"{col}_1"] = df1.loc[match_df["idx1"], col].to_numpy()
        match_df[f"{col}_2"] = df2.loc[match_df["idx2"], col].to_numpy()

    return match_df.reset_index(drop=True)
