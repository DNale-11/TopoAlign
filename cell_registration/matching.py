"""Cell matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    )
    # Weight for spatial proximity; 0 disables spatial cue. Uses normalized positions (pos_x_norm/pos_y_norm).
    position_weight: float = 1.0
    top_k: int = 10


def _standardize_features(
    df1: pd.DataFrame, df2: pd.DataFrame, feature_columns: Tuple[str, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    combined = pd.concat(
        [df1.loc[:, feature_columns], df2.loc[:, feature_columns]], axis=0, ignore_index=True
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]
    return f1, f2


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns {missing}. "
            "Compute features first via `compute_cell_features` (regionprops-based)."
        )


def greedy_match_cells(df1: pd.DataFrame, df2: pd.DataFrame, config: MatchingConfig) -> pd.DataFrame:
    """
    Greedy 1-to-1 matching between two sets of cells using feature similarity.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Feature tables with required columns.
    config : MatchingConfig
        Matching configuration.

    Returns
    -------
    pd.DataFrame
        Matches with indices and distances.
    """
    feat_cols = config.feature_columns
    _ensure_columns(df1, feat_cols)
    _ensure_columns(df2, feat_cols)
    f1, f2 = _standardize_features(df1, df2, feat_cols)

    # Feature distance
    feat_dist = np.linalg.norm(f1[:, None, :] - f2[None, :, :], axis=2)

    # Spatial distance on normalized coordinates (stable across image sizes; penalizes far-apart cells).
    if config.position_weight > 0:
        _ensure_columns(df1, ("pos_x_norm", "pos_y_norm"))
        _ensure_columns(df2, ("pos_x_norm", "pos_y_norm"))
        coords1 = df1[["pos_x_norm", "pos_y_norm"]].to_numpy()
        coords2 = df2[["pos_x_norm", "pos_y_norm"]].to_numpy()
        pos_dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
    else:
        pos_dist = 0

    dist_matrix = feat_dist + config.position_weight * pos_dist

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
        matches.append((i, j, d))
        used_1.add(i)
        used_2.add(j)
        if len(matches) >= config.top_k:
            break

    match_df = pd.DataFrame(matches, columns=["idx1", "idx2", "distance"])
    match_df["cell_id_1"] = df1.iloc[match_df["idx1"]]["cell_id"].to_numpy()
    match_df["cell_id_2"] = df2.iloc[match_df["idx2"]]["cell_id"].to_numpy()

    # Optionally include features for inspection
    for col in feat_cols:
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
    Useful to guarantee an even spatial distribution (e.g., 6 pairs per 3x3 patch -> 54 total).
    """
    if top_k_per_patch < 1:
        raise ValueError("top_k_per_patch must be >= 1.")

    feat_cols = config.feature_columns
    _ensure_columns(df1, feat_cols)
    _ensure_columns(df2, feat_cols)
    _ensure_columns(df1, ("patch_x", "patch_y"))
    _ensure_columns(df2, ("patch_x", "patch_y"))

    # Standardize across the whole image so feature scales are consistent between patches.
    f1, f2 = _standardize_features(df1, df2, feat_cols)

    rows: list[tuple[int, int, float, int, int]] = []
    patches = sorted(set(zip(df1["patch_x"], df1["patch_y"])) & set(zip(df2["patch_x"], df2["patch_y"])))

    for px, py in patches:
        idxs1 = df1.index[(df1["patch_x"] == px) & (df1["patch_y"] == py)].to_list()
        idxs2 = df2.index[(df2["patch_x"] == px) & (df2["patch_y"] == py)].to_list()
        if not idxs1 or not idxs2:
            continue

        dist_matrix = np.linalg.norm(f1[idxs1][:, None, :] - f2[idxs2][None, :, :], axis=2)
        if config.position_weight > 0:
            coords1 = df1.loc[idxs1, ["pos_x_norm", "pos_y_norm"]].to_numpy()
            coords2 = df2.loc[idxs2, ["pos_x_norm", "pos_y_norm"]].to_numpy()
            pos_dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
        else:
            pos_dist = 0

        dist_matrix = dist_matrix + config.position_weight * pos_dist

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

    for col in feat_cols:
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

    feat_cols = config.feature_columns
    _ensure_columns(df1, feat_cols)
    _ensure_columns(df2, feat_cols)

    rows: list[dict] = []
    clusters = sorted(set(df1[cluster_col]) & set(df2[cluster_col]))
    for cluster in clusters:
        idxs1 = df1.index[df1[cluster_col] == cluster].to_list()
        idxs2 = df2.index[df2[cluster_col] == cluster].to_list()
        if not idxs1 or not idxs2:
            continue

        # Work on local copies to reuse greedy_match_cells while keeping global index mapping.
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
        for cluster, group in match_df.groupby(cluster_col):
            kept_groups.append(group.sort_values("distance", ascending=True).head(top_k_per_cluster))
        match_df = pd.concat(kept_groups, ignore_index=True)

    for col in feat_cols:
        match_df[f"{col}_1"] = df1.loc[match_df["idx1"], col].to_numpy()
        match_df[f"{col}_2"] = df2.loc[match_df["idx2"], col].to_numpy()

    return match_df.reset_index(drop=True)
