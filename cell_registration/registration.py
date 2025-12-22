"""Rigid/similarity transform estimation from matched cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class RigidTransform:
    """Rigid (or similarity) transform in 2D or 3D."""

    rotation: np.ndarray  # shape (2, 2) or (3, 3)
    translation: np.ndarray  # shape (2,) or (3,)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.rotation, self.translation


def estimate_rigid_transform_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_df: pd.DataFrame,
    use_scale: bool = False,  # Ignored for rigid
) -> RigidTransform:
    """
    Estimate a rigid transform (Rotation + Translation) using matched cell centroids.
    Supports 2D and 3D.
    """
    if match_df.empty:
        raise ValueError("No matches provided to estimate transform.")

    # Detect dimensionality
    if "centroid_z" in df1.columns and "centroid_z" in df2.columns:
        dims = ["centroid_z", "centroid_y", "centroid_x"]
        is_3d = True
    else:
        dims = ["centroid_y", "centroid_x"] # Note: kept y, x order to match
        is_3d = False

    # Extract points
    # We must ensure we pull them in consistent order.
    # Usually [z, y, x] or [y, x].
    # Let's standardize on [z, y, x] for 3D and [y, x] for 2D?
    # Actually, downstream RANSAC usually expects (N, D) arrays.

    pts_a_list = []
    pts_b_list = []

    # Efficient extraction
    idx1 = match_df["idx1"].values
    idx2 = match_df["idx2"].values

    A = df1.iloc[idx1][dims].to_numpy(dtype=np.float64)
    B = df2.iloc[idx2][dims].to_numpy(dtype=np.float64)

    # We want to find R, t such that A ~= B @ R.T + t  (skimage convention)
    # OR A ~= R @ B + t (standard math convention).
    # `skimage.transform.EuclideanTransform` does:
    #   coord_dst = coord_src @ rotation + translation
    # Wait, skimage EuclideanTransform matrix is 3x3 (for 2D), acting on [x, y, 1].
    # [x', y', 1] = [x, y, 1] @ M.T  => x' = x*cos - y*sin + tx

    # Kabsch algorithm usually solves B @ R + t = A  (where row vectors)
    # or R @ B_col + t = A_col.

    # Let's align B to A.
    # Center the points
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    # Covariance matrix H = BB.T @ AA
    H = BB.T @ AA

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # R = V @ U.T
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # t = centroid_A - centroid_B @ R
    # (Note: if using row vectors x' = xR + t, then t = centroid_A - centroid_B @ R)
    # If using column vectors x' = Rx + t, then t = centroid_A - R @ centroid_B

    # Let's stick to row vector convention used by skimage: dst = src @ R + t
    # where R is technically the transpose of the standard rotation matrix if src is row vector.
    # Wait, skimage documentation says:
    # "The transformation is applied as: [x, y, 1] = [x, y, 1] @ matrix.T"
    # which effectively is col_vec' = matrix @ col_vec.

    # Let's define our RigidTransform class to store standard R (d x d) and t (d).
    # And we assume application is: x_new = (R @ x_old.T).T + t

    # So if we want B_aligned ~ A, we want: (R @ B.T).T + t ~ A
    # => B @ R.T + t ~ A

    # In Kabsch, if we use H = BB.T @ AA, then R (calculated as Vt.T @ U.T) satisfies:
    # A_centered ~ B_centered @ R.T
    # Or is it B_centered @ R?
    # Let's verify.
    # H = sum( b_i.T * a_i ) (if column vectors)
    # H = BB.T @ AA (if BB and AA are N x D matrices).
    # Then R optimal is such that A = R_opt @ B (for column vectors).
    # So A.T = B.T @ R_opt.T.
    # So for row vectors AA ~ BB @ R_opt.T.
    # My derivation above R = Vt.T @ U.T gives the rotation matrix for column vectors.

    # So if I want `x_new = (R @ x_old.T).T + t`, then R is the column-vector rotation matrix.
    # So I can use R directly in my class.

    t = centroid_A - (R @ centroid_B.T).T
    # t should be shape (D,).
    # centroid_A is (D,), (R @ centroid_B.T).T is (D,).

    return RigidTransform(rotation=R, translation=t)


def refine_with_neighbors(*args, **kwargs):
    """Placeholder for future neighborhood-based refinement."""
    raise NotImplementedError("Neighborhood-based refinement is not implemented yet.")
