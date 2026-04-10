"""
Robust global alignment using RANSAC on feature-based candidate matches.

Uses a translation-only model (no rotation) since cell imaging rounds
are assumed to differ only by a small shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac
from skimage.transform import EuclideanTransform


class TranslationTransform:
    """Translation-only 2D transform compatible with skimage RANSAC.

    This model estimates only (tx, ty) with zero rotation, which is more
    robust than ``EuclideanTransform`` when feature-based candidate matches
    are noisy and could otherwise lead RANSAC to fit spurious large rotations.
    """

    def __init__(self):
        self._translation = np.zeros(2, dtype=float)

    # -- properties compatible with EuclideanTransform --
    @property
    def rotation(self) -> float:
        """Always 0 – no rotation is estimated."""
        return 0.0

    @property
    def translation(self) -> np.ndarray:
        return self._translation.copy()

    @property
    def params(self) -> np.ndarray:
        """3×3 homogeneous matrix (translation only)."""
        m = np.eye(3, dtype=float)
        m[0, 2] = self._translation[0]
        m[1, 2] = self._translation[1]
        return m

    # -- skimage RANSAC interface --
    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """Estimate translation as the mean of (dst − src)."""
        self._translation = np.mean(dst - src, axis=0)
        return True

    def residuals(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Per-point Euclidean residual after applying the translation."""
        transformed = src + self._translation
        return np.sqrt(np.sum((transformed - dst) ** 2, axis=1))

    # -- callable interface --
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Apply translation to an (N, 2) coordinate array."""
        return np.asarray(coords, dtype=float) + self._translation


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
) -> TranslationTransform | None:
    """
    Estimate a global **translation-only** transform aligning df2 to df1.

    Uses RANSAC with a ``TranslationTransform`` model so that noisy
    feature candidates cannot produce spurious large rotations.

    Returns None if registration fails.
    """
    if df1.empty or df2.empty:
        return None

    # Get putative matches based on morphology
    src, dst = get_feature_candidates(df1, df2, feature_columns, top_k=top_k_candidates)

    if len(src) < ransac_min_samples:
        return None

    # Run RANSAC with translation-only model (2 DOF: tx, ty)
    try:
        np.random.seed(42)
        model, inliers = ransac(
            (src, dst),
            TranslationTransform,
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
    transform,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y"
) -> pd.DataFrame:
    """
    Apply a transform (TranslationTransform or EuclideanTransform) to
    feature coordinates and return a copy.
    """
    df_out = df.copy()
    coords = df_out[[x_col, y_col]].to_numpy()

    # transform(coords) applies the transformation
    aligned = transform(coords)

    df_out[x_col] = aligned[:, 0]
    df_out[y_col] = aligned[:, 1]

    return df_out

