"""Rigid/similarity transform estimation from matched cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform, estimate_transform


@dataclass
class RigidTransform:
    """2D rigid/similarity transform stored as a linear part plus translation."""

    rotation: np.ndarray  # shape (2, 2); may include a uniform scale factor
    translation: np.ndarray  # shape (2,)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.rotation, self.translation

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(3, dtype=float)
        matrix[:2, :2] = np.asarray(self.rotation, dtype=float)
        matrix[:2, 2] = np.asarray(self.translation, dtype=float)
        return matrix

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "RigidTransform":
        params = np.asarray(matrix, dtype=float)
        if params.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 homogeneous matrix, got {params.shape}.")
        return cls(rotation=params[:2, :2].copy(), translation=params[:2, 2].copy())

    def as_affine_transform(self) -> AffineTransform:
        return AffineTransform(matrix=self.matrix)


def build_inverse_affine_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> AffineTransform:
    """
    Build an inverse affine transform for image warping from a forward point transform.

    The forward convention throughout the package is:
        target_xy = rotation @ source_xy + translation
    """
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = np.asarray(rotation, dtype=float)
    matrix[:2, 2] = np.asarray(translation, dtype=float)
    return AffineTransform(matrix=np.linalg.inv(matrix))


def _matched_points(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if match_df.empty:
        raise ValueError("No matches provided to estimate transform.")

    idx1 = match_df["idx1"].to_numpy(dtype=int, copy=False)
    idx2 = match_df["idx2"].to_numpy(dtype=int, copy=False)
    pts_fixed = df1.iloc[idx1][["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64, copy=True)
    pts_moving = df2.iloc[idx2][["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64, copy=True)
    return pts_fixed, pts_moving


def _translation_only_transform(pts_fixed: np.ndarray, pts_moving: np.ndarray) -> RigidTransform:
    translation = pts_fixed.mean(axis=0) - pts_moving.mean(axis=0)
    return RigidTransform(rotation=np.eye(2, dtype=float), translation=translation.astype(float, copy=False))


def _estimate_transform_from_points(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    *,
    use_scale: bool,
) -> RigidTransform:
    if len(pts_fixed) == 0:
        raise ValueError("No points provided to estimate transform.")
    if len(pts_fixed) < 2:
        return _translation_only_transform(pts_fixed, pts_moving)

    method = "similarity" if use_scale else "euclidean"
    try:
        model = estimate_transform(method, pts_moving, pts_fixed)
        params = np.asarray(model.params, dtype=float)
        if not np.all(np.isfinite(params)):
            raise ValueError("Estimated transform contains non-finite values.")
        return RigidTransform.from_matrix(params)
    except Exception:
        return _translation_only_transform(pts_fixed, pts_moving)


def estimate_rigid_transform_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_df: pd.DataFrame,
    use_scale: bool = True,
) -> RigidTransform:
    """
    Estimate a global transform from matched cell centroids.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Feature tables containing centroid_x and centroid_y.
    match_df : pd.DataFrame
        Output of greedy_match_cells with idx1 and idx2 columns.
    use_scale : bool, optional
        If True, fit a similarity transform (rotation + translation + uniform scale).
        If False, fit a Euclidean transform (rotation + translation only).

    Returns
    -------
    RigidTransform
        Estimated transform.
    """
    pts_fixed, pts_moving = _matched_points(df1, df2, match_df)
    return _estimate_transform_from_points(pts_fixed, pts_moving, use_scale=use_scale)


def estimate_rigid_transform_from_matches_ransac(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_df: pd.DataFrame,
    *,
    max_trials: int = 1000,
    residual_threshold: float = 2.0,
    min_inliers: int = 3,
    use_scale: bool = True,
) -> tuple[RigidTransform, np.ndarray]:
    """Estimate a robust transform with RANSAC from matched cell centroids."""
    pts_fixed, pts_moving = _matched_points(df1, df2, match_df)
    n_matches = len(match_df)
    min_samples = 2
    if n_matches < max(min_inliers, min_samples):
        transform = _estimate_transform_from_points(pts_fixed, pts_moving, use_scale=use_scale)
        return transform, np.ones(n_matches, dtype=bool)

    model_class = SimilarityTransform if use_scale else EuclideanTransform
    method = "similarity" if use_scale else "euclidean"
    try:
        model, inliers = ransac(
            (pts_moving, pts_fixed),
            model_class,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
    except Exception:
        model = None
        inliers = None

    if model is None or inliers is None or int(np.count_nonzero(inliers)) < max(min_inliers, min_samples):
        transform = _estimate_transform_from_points(pts_fixed, pts_moving, use_scale=use_scale)
        return transform, np.ones(n_matches, dtype=bool)

    try:
        refit = estimate_transform(method, pts_moving[inliers], pts_fixed[inliers])
        params = np.asarray(refit.params, dtype=float)
        if not np.all(np.isfinite(params)):
            raise ValueError("Refit transform contains non-finite values.")
        transform = RigidTransform.from_matrix(params)
    except Exception:
        transform = RigidTransform.from_matrix(np.asarray(model.params, dtype=float))
    return transform, np.asarray(inliers, dtype=bool)


def refine_with_neighbors(*args, **kwargs):
    """Placeholder for future neighborhood-based refinement."""
    raise NotImplementedError("Neighborhood-based refinement is not implemented yet.")
