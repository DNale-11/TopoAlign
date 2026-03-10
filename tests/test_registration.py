import numpy as np
import pandas as pd

from cell_registration.registration import (
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
)


def _make_features(points: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, len(points) + 1),
            "centroid_x": points[:, 0],
            "centroid_y": points[:, 1],
        }
    )


def _similarity_matrix(scale: float, theta_deg: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [scale * c, -scale * s, tx],
            [scale * s, scale * c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_estimate_rigid_transform_from_matches_recovers_similarity() -> None:
    moving = np.array(
        [
            [10.0, 15.0],
            [30.0, 12.0],
            [45.0, 32.0],
            [18.0, 50.0],
            [60.0, 44.0],
        ],
        dtype=float,
    )
    matrix = _similarity_matrix(scale=1.03, theta_deg=7.5, tx=12.0, ty=-8.0)
    fixed = (matrix[:2, :2] @ moving.T).T + matrix[:2, 2]

    df1 = _make_features(fixed)
    df2 = _make_features(moving)
    matches = pd.DataFrame({"idx1": np.arange(len(moving)), "idx2": np.arange(len(moving))})

    transform = estimate_rigid_transform_from_matches(df1, df2, matches, use_scale=True)

    np.testing.assert_allclose(transform.rotation, matrix[:2, :2], atol=1e-6)
    np.testing.assert_allclose(transform.translation, matrix[:2, 2], atol=1e-6)


def test_estimate_rigid_transform_from_matches_ransac_rejects_outlier() -> None:
    moving = np.array(
        [
            [5.0, 8.0],
            [16.0, 10.0],
            [28.0, 26.0],
            [12.0, 32.0],
            [35.0, 38.0],
            [45.0, 18.0],
        ],
        dtype=float,
    )
    matrix = _similarity_matrix(scale=0.98, theta_deg=-4.0, tx=6.5, ty=9.0)
    fixed = (matrix[:2, :2] @ moving.T).T + matrix[:2, 2]

    df1 = _make_features(np.vstack([fixed, [[500.0, -300.0]]]))
    df2 = _make_features(np.vstack([moving, [[-200.0, 700.0]]]))
    matches = pd.DataFrame({"idx1": np.arange(len(df1)), "idx2": np.arange(len(df2))})

    transform, inliers = estimate_rigid_transform_from_matches_ransac(
        df1,
        df2,
        matches,
        residual_threshold=0.75,
        max_trials=500,
        min_inliers=4,
        use_scale=True,
    )

    assert int(inliers.sum()) == len(moving)
    np.testing.assert_allclose(transform.rotation, matrix[:2, :2], atol=1e-3)
    np.testing.assert_allclose(transform.translation, matrix[:2, 2], atol=1e-3)
