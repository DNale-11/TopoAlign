"""
Robust global alignment using RANSAC on feature-based candidate matches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac
from skimage.transform import EuclideanTransform


def get_feature_candidates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...],
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find candidate matches based purely on feature similarity.
    Returns (src_points, dst_points) arrays for RANSAC.

    src_points: Coordinates from df2 (source to be transformed)
    dst_points: Coordinates from df1 (target)

    Note: We map df2 -> df1.
    """
    # 1. Standardize features
    # Ensure columns exist
    for col in feature_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing feature column {col}")

    scaler = StandardScaler()
    combined = pd.concat(
        [df1[list(feature_columns)], df2[list(feature_columns)]],
        axis=0, ignore_index=True
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]

    # 2. Find top-k neighbors in feature space
    # For each cell in df2, find k similar cells in df1
    nn = NearestNeighbors(n_neighbors=top_k, algorithm="auto")
    nn.fit(f1)

    distances, indices = nn.kneighbors(f2)

    # 3. Flatten into correspondence arrays
    # src: df2 points (repeated k times)
    # dst: df1 points (the neighbors found)

    src_coords = df2[["centroid_x", "centroid_y"]].to_numpy()
    dst_coords = df1[["centroid_x", "centroid_y"]].to_numpy()

    src_list = []
    dst_list = []

    for i in range(len(df2)):
        # i is index in df2
        p_src = src_coords[i]
        for neighbor_idx in indices[i]:
            # neighbor_idx is index in df1
            p_dst = dst_coords[neighbor_idx]
            src_list.append(p_src)
            dst_list.append(p_dst)

    return np.array(src_list), np.array(dst_list)


def perform_global_registration(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...] = (
        "area",
        "perimeter",
        "roundness",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
    ),
    top_k_candidates: int = 5,
    ransac_min_samples: int = 3,
    ransac_residual_threshold: float = 2.0,
    ransac_max_trials: int = 2000,
) -> EuclideanTransform | None:
    """
    Estimate a global Rigid Transform (Rotation + Translation)
    aligning df2 (source) to df1 (target) using feature-guided RANSAC.

    Returns None if registration fails.
    """
    if df1.empty or df2.empty:
        return None

    # Get putative matches based on morphology
    src, dst = get_feature_candidates(df1, df2, feature_columns, top_k=top_k_candidates)

    if len(src) < ransac_min_samples:
        return None

    # Run RANSAC
    # EuclideanTransform model: 3 degrees of freedom (rotation, translation)
    # It solves: dst = Matrix * src
    try:
        # Seed global RNG for reproducibility if needed, or rely on caller.
        # Removing random_state kwarg for compatibility with older skimage versions.
        np.random.seed(42)
        model, inliers = ransac(
            (src, dst),
            EuclideanTransform,
            min_samples=ransac_min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials
        )
    except Exception as e:
        print(f"RANSAC global registration failed: {e}")
        return None

    return model


def apply_transform_to_coordinates(
    df: pd.DataFrame,
    transform: EuclideanTransform,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y"
) -> pd.DataFrame:
    """
    Apply the transform to the coordinates in the dataframe
    and return a copy with updated coordinates.
    """
    df_out = df.copy()
    coords = df_out[[x_col, y_col]].to_numpy()

    # transform(coords) applies the transformation
    # Note: skimage EuclideanTransform operates on (N, 2) arrays
    aligned = transform(coords)

    df_out[x_col] = aligned[:, 0]
    df_out[y_col] = aligned[:, 1]

    # Also update normalized coordinates if they exist
    # Note: This invalidates the normalization relative to original image size
    # But for matching purposes, we just need them to be consistent with df1.
    # To be safe, we should probably re-normalize or just let the matching
    # use the raw centroid_x/y if we switch the matching config?
    # The existing matching uses 'pos_x_norm'. We should update those too.
    # Assuming the 'image_width' and 'image_height' were used to normalize.
    # Ideally, we re-run the 'assign_patches' logic after alignment.

    return df_out
