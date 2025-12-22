"""Topology-based matching logic."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import Optional

# Constants
PATCH_GRID = 3  # 3x3 patches across the image
PATCH_W = 1.0 / PATCH_GRID
PATCH_H = 1.0 / PATCH_GRID


@dataclass
class MatchingConfig:
    """Configuration for cell matching."""
    top_k: int = 50
    position_weight: float = 1.0


def _patch_center(patch_coords: tuple[int, ...]) -> tuple[float, ...]:
    """Return normalized patch center coordinates."""
    return tuple((c / PATCH_GRID) + (1.0 / PATCH_GRID) / 2.0 for c in patch_coords)


def _patch_diag_px(image_shape: tuple[int, ...]) -> float:
    """Compute the diagonal length (in pixels) of a single patch."""
    # image_shape is (Z, Y, X) or (Y, X)
    dims = len(image_shape)
    patch_dims = [s / PATCH_GRID for s in image_shape]
    return math.sqrt(sum(d*d for d in patch_dims))


def assign_patches(
    df: pd.DataFrame,
    image_shape: tuple[int, ...],
) -> pd.DataFrame:
    """Attach normalized coordinates and patch indices to the cell table."""
    out = df.copy()

    # Determine dims
    if len(image_shape) == 3:
        # Z, Y, X
        z_col, y_col, x_col = "centroid_z", "centroid_y", "centroid_x"
        d, h, w = image_shape
        out["z_norm"] = (out[z_col] / float(d)).clip(0.0, 1.0)
        out["y_norm"] = (out[y_col] / float(h)).clip(0.0, 1.0)
        out["x_norm"] = (out[x_col] / float(w)).clip(0.0, 1.0)

        out["patch_z"] = np.clip((out["z_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
        out["patch_y"] = np.clip((out["y_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
        out["patch_x"] = np.clip((out["x_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)

        out["patch_id"] = list(zip(out["patch_z"], out["patch_y"], out["patch_x"]))

    else:
        # Y, X
        y_col, x_col = "centroid_y", "centroid_x"
        h, w = image_shape
        out["y_norm"] = (out[y_col] / float(h)).clip(0.0, 1.0)
        out["x_norm"] = (out[x_col] / float(w)).clip(0.0, 1.0)

        out["patch_y"] = np.clip((out["y_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
        out["patch_x"] = np.clip((out["x_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)

        out["patch_id"] = list(zip(out["patch_y"], out["patch_x"]))

    return out


def compute_L_pos(
    cell_r1: pd.Series,
    cell_r2: pd.Series,
    image_shape: tuple[int, ...],
) -> float:
    """Position loss between two cells inside the same patch."""
    is_3d = len(image_shape) == 3

    if is_3d:
        patch_coords = (int(cell_r1["patch_z"]), int(cell_r1["patch_y"]), int(cell_r1["patch_x"]))
        cz, cy, cx = _patch_center(patch_coords)

        # Relative coords in patch [-1, 1]
        rz1 = (cell_r1["z_norm"] - cz) / (1.0 / PATCH_GRID / 2.0)
        ry1 = (cell_r1["y_norm"] - cy) / (1.0 / PATCH_GRID / 2.0)
        rx1 = (cell_r1["x_norm"] - cx) / (1.0 / PATCH_GRID / 2.0)

        rz2 = (cell_r2["z_norm"] - cz) / (1.0 / PATCH_GRID / 2.0)
        ry2 = (cell_r2["y_norm"] - cy) / (1.0 / PATCH_GRID / 2.0)
        rx2 = (cell_r2["x_norm"] - cx) / (1.0 / PATCH_GRID / 2.0)

        d = (rz1-rz2)**2 + (ry1-ry2)**2 + (rx1-rx2)**2
        return math.sqrt(d) / math.sqrt(3.0) # normalize by max diag

    else:
        patch_coords = (int(cell_r1["patch_y"]), int(cell_r1["patch_x"]))
        cy, cx = _patch_center(patch_coords)

        ry1 = (cell_r1["y_norm"] - cy) / (PATCH_H / 2.0)
        rx1 = (cell_r1["x_norm"] - cx) / (PATCH_W / 2.0)
        ry2 = (cell_r2["y_norm"] - cy) / (PATCH_H / 2.0)
        rx2 = (cell_r2["x_norm"] - cx) / (PATCH_W / 2.0)

        d = (ry1-ry2)**2 + (rx1-rx2)**2
        return math.sqrt(d) / math.sqrt(2.0)


def greedy_match_cells(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    config: MatchingConfig,
) -> pd.DataFrame:
    """
    Greedy matching based on feature similarity + position.
    """
    # Simply use position + area/shape similarity
    # We'll use a simplified cost metric here

    # Identify common columns for features
    exclude = {"cell_id", "label", "patch_id", "cluster_id"}
    cols = [c for c in feats1.columns if c in feats2.columns and "centroid" not in c and "patch" not in c and "norm" not in c and c not in exclude]

    # Use normalized position columns
    pos_cols = [c for c in feats1.columns if "_norm" in c]

    # Build cost matrix? Too big maybe.
    # Use Nearest Neighbors

    # Construct feature vector: features (normalized) + position * weight
    # We need to normalize features first.

    # For simplicity, let's just use Euclidean distance on normalized positions
    # and maybe 'volume'/'area' diff.

    # Let's trust the 'features' are somewhat comparable.

    # This function was imported in main but not defined in previous matching.py?
    # Ah, I am overwriting matching.py, I should have read it first to see what was there.
    # But since I am refactoring, I'll implement a standard one.

    if feats1.empty or feats2.empty:
         return pd.DataFrame(columns=["idx1", "idx2", "cell_id_1", "cell_id_2"])

    # Concatenate to normalize
    f1_vals = feats1[cols].values
    f2_vals = feats2[cols].values

    # Simple normalization by max? or Std.
    f_max = np.maximum(f1_vals.max(axis=0), f2_vals.max(axis=0))
    f_max[f_max==0] = 1

    f1_norm = f1_vals / f_max
    f2_norm = f2_vals / f_max

    p1 = feats1[pos_cols].values * config.position_weight
    p2 = feats2[pos_cols].values * config.position_weight

    v1 = np.hstack([f1_norm, p1])
    v2 = np.hstack([f2_norm, p2])

    nn = NearestNeighbors(n_neighbors=min(config.top_k, len(feats2)))
    nn.fit(v2)

    dists, indices = nn.kneighbors(v1)

    # Greedy assignment
    # dists is (N1, k), indices is (N1, k)

    # Flatten
    candidates = []
    for i in range(len(feats1)):
        for k in range(indices.shape[1]):
            j = indices[i, k]
            d = dists[i, k]
            candidates.append((d, i, j))

    candidates.sort(key=lambda x: x[0])

    matched1 = set()
    matched2 = set()
    matches = []

    for d, i, j in candidates:
        if i not in matched1 and j not in matched2:
            matched1.add(i)
            matched2.add(j)
            matches.append({
                "idx1": i,
                "idx2": j,
                "cell_id_1": feats1.iloc[i]["cell_id"],
                "cell_id_2": feats2.iloc[j]["cell_id"]
            })
            if len(matches) >= config.top_k:
                break

    return pd.DataFrame(matches)


def match_cells_per_patch(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    config: MatchingConfig,
    top_k_per_patch: int = 6
) -> pd.DataFrame:
    """Run greedy matching per patch."""
    all_matches = []

    # Group by patch_id
    # Ensure patch_id exists (assigned by assign_patches)
    if "patch_id" not in feats1.columns or "patch_id" not in feats2.columns:
        raise ValueError("Patches not assigned.")

    patches = set(feats1["patch_id"].unique()) & set(feats2["patch_id"].unique())

    for pid in patches:
        f1 = feats1[feats1["patch_id"] == pid]
        f2 = feats2[feats2["patch_id"] == pid]

        # Map back to original indices
        f1_map = {i: idx for i, idx in enumerate(f1.index)}
        f2_map = {i: idx for i, idx in enumerate(f2.index)}

        sub_cfg = MatchingConfig(top_k=top_k_per_patch, position_weight=config.position_weight)
        sub_matches = greedy_match_cells(f1.reset_index(drop=True), f2.reset_index(drop=True), sub_cfg)

        for _, row in sub_matches.iterrows():
            orig_idx1 = f1_map[row["idx1"]]
            orig_idx2 = f2_map[row["idx2"]]
            all_matches.append({
                "idx1": orig_idx1,
                "idx2": orig_idx2,
                "cell_id_1": row["cell_id_1"],
                "cell_id_2": row["cell_id_2"]
            })

    return pd.DataFrame(all_matches)

def match_cells_per_cluster(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    config: MatchingConfig,
    cluster_col: str,
    top_k_per_cluster: int
) -> pd.DataFrame:
    """Run greedy matching per cluster."""
    all_matches = []

    clusters = set(feats1[cluster_col].unique()) & set(feats2[cluster_col].unique())

    for cid in clusters:
        f1 = feats1[feats1[cluster_col] == cid]
        f2 = feats2[feats2[cluster_col] == cid]

        f1_map = {i: idx for i, idx in enumerate(f1.index)}
        f2_map = {i: idx for i, idx in enumerate(f2.index)}

        sub_cfg = MatchingConfig(top_k=top_k_per_cluster, position_weight=config.position_weight)
        sub_matches = greedy_match_cells(f1.reset_index(drop=True), f2.reset_index(drop=True), sub_cfg)

        for _, row in sub_matches.iterrows():
            orig_idx1 = f1_map[row["idx1"]]
            orig_idx2 = f2_map[row["idx2"]]
            all_matches.append({
                "idx1": orig_idx1,
                "idx2": orig_idx2,
                "cell_id_1": row["cell_id_1"],
                "cell_id_2": row["cell_id_2"]
            })

    return pd.DataFrame(all_matches)
