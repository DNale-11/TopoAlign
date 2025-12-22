"""
Robust global alignment using RANSAC on feature-based candidate matches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac

from .registration import RigidTransform, estimate_rigid_transform_from_matches


class RigidTransformModel:
    """
    Model class for scikit-image RANSAC.
    Estimates a RigidTransform (Rotation + Translation) for N-dim.
    """
    def __init__(self):
        self.params = None

    def estimate(self, src, dst):
        """
        Estimate the transformation from src to dst.
        src, dst: (N, D) arrays
        """
        # We can reuse our estimate_rigid_transform_from_matches function
        # but we need to wrap inputs into dataframes or just extract the logic.
        # Let's extract the pure numpy logic here to avoid overhead.

        if len(src) < 2: # Need at least some points. 3 for 3D usually, 2 for 2D is enough if no scale.
             return False

        centroid_src = src.mean(axis=0)
        centroid_dst = dst.mean(axis=0)

        src_centered = src - centroid_src
        dst_centered = dst - centroid_dst

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_dst - (R @ centroid_src.T).T

        self.params = (R, t)
        return True

    def residuals(self, src, dst):
        """
        Calculate residuals for each point.
        """
        R, t = self.params
        src_transformed = (R @ src.T).T + t
        return np.linalg.norm(src_transformed - dst, axis=1)

    @property
    def rotation(self):
        return self.params[0]

    @property
    def translation(self):
        return self.params[1]


def get_feature_candidates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...],
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find candidate matches based purely on feature similarity.
    Returns (src_points, dst_points) arrays for RANSAC.
    """
    # Filter available columns
    valid_cols = [c for c in feature_columns if c in df1.columns and c in df2.columns]
    if not valid_cols:
         # Fallback to just area/volume if nothing else
         if "volume" in df1.columns and "volume" in df2.columns:
             valid_cols = ["volume"]
         elif "area" in df1.columns and "area" in df2.columns:
             valid_cols = ["area"]
         else:
             raise ValueError("No matching feature columns found.")

    scaler = StandardScaler()
    combined = pd.concat(
        [df1[valid_cols], df2[valid_cols]],
        axis=0, ignore_index=True
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]

    nn = NearestNeighbors(n_neighbors=top_k, algorithm="auto")
    nn.fit(f1)

    distances, indices = nn.kneighbors(f2)

    # Coordinates
    if "centroid_z" in df2.columns:
        cols = ["centroid_z", "centroid_y", "centroid_x"]
    else:
        cols = ["centroid_y", "centroid_x"]

    src_coords = df2[cols].to_numpy()
    dst_coords = df1[cols].to_numpy()

    src_list = []
    dst_list = []

    for i in range(len(df2)):
        p_src = src_coords[i]
        for neighbor_idx in indices[i]:
            p_dst = dst_coords[neighbor_idx]
            src_list.append(p_src)
            dst_list.append(p_dst)

    return np.array(src_list), np.array(dst_list)


def perform_global_registration(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...] = (
        "volume",
        "area",
        "major_axis_length",
        "minor_axis_length",
        "solidity", # Note: eccentricity/perimeter might be missing in 3D
    ),
    top_k_candidates: int = 5,
    ransac_min_samples: int = 4, # Safer for 3D
    ransac_residual_threshold: float = 2.0,
    ransac_max_trials: int = 2000,
) -> RigidTransform | None:
    """
    Estimate a global Rigid Transform (Rotation + Translation)
    aligning df2 (source) to df1 (target) using feature-guided RANSAC.
    """
    if df1.empty or df2.empty:
        return None

    src, dst = get_feature_candidates(df1, df2, feature_columns, top_k=top_k_candidates)

    if len(src) < ransac_min_samples:
        return None

    try:
        np.random.seed(42)
        model, inliers = ransac(
            (src, dst),
            RigidTransformModel,
            min_samples=ransac_min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials
        )
    except Exception as e:
        print(f"RANSAC global registration failed: {e}")
        return None

    if model is None or model.params is None:
        return None

    return RigidTransform(rotation=model.rotation, translation=model.translation)


def apply_transform_to_coordinates(
    df: pd.DataFrame,
    transform: RigidTransform,
) -> pd.DataFrame:
    """
    Apply the transform to the coordinates in the dataframe.
    """
    df_out = df.copy()

    if "centroid_z" in df_out.columns:
        cols = ["centroid_z", "centroid_y", "centroid_x"]
    else:
        cols = ["centroid_y", "centroid_x"]

    coords = df_out[cols].to_numpy()

    # Apply x_new = (R @ x_old.T).T + t
    aligned = (transform.rotation @ coords.T).T + transform.translation

    for i, col in enumerate(cols):
        df_out[col] = aligned[:, i]

    return df_out
