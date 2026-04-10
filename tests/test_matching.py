from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.transform import AffineTransform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cell_registration import matching


def _make_feature_table() -> pd.DataFrame:
    coords = []
    cell_id = 0
    rows = []
    for patch_y in range(3):
        for patch_x in range(3):
            for dx, dy in ((0.0, 0.0), (12.0, 8.0)):
                x = 50.0 + patch_x * 100.0 + dx
                y = 40.0 + patch_y * 100.0 + dy
                rows.append(
                    {
                        "cell_id": cell_id,
                        "centroid_x": x,
                        "centroid_y": y,
                        "pos_x_norm": x / 400.0,
                        "pos_y_norm": y / 400.0,
                        "area": 100.0,
                        "perimeter": 40.0,
                        "roundness": 0.8,
                        "eccentricity": 0.4,
                        "solidity": 0.9,
                        "major_axis_length": 12.0,
                        "minor_axis_length": 8.0,
                        "aspect_ratio": 1.5,
                        "elongation": 1.0 - (8.0 / 12.0),
                        "equivalent_diameter": np.sqrt(4.0 * 100.0 / np.pi),
                        "nn_dist_1": 1.0,
                        "nn_dist_2": 1.0,
                        "nn_dist_3": 1.0,
                        "local_density": 1.0,
                        "neighbor_area_ratio_mean": 1.0,
                        "neighbor_roundness_mean": 0.8,
                    }
                )
                cell_id += 1
    return pd.DataFrame(rows)


def _coarse_matches_stub(*_args, **_kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx1": np.arange(18, dtype=int),
            "idx2": np.arange(18, dtype=int),
            "distance": np.full(18, 0.1, dtype=float),
        }
    )


def _fine_matches_stub(*_args, **_kwargs) -> pd.DataFrame:
    return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])


def test_two_stage_rejects_weak_coarse_transform(monkeypatch):
    feats1 = _make_feature_table()
    feats2 = _make_feature_table()
    candidate_transform = AffineTransform(translation=(25.0, -10.0))

    monkeypatch.setattr(matching, "match_cells_per_patch", _coarse_matches_stub)
    monkeypatch.setattr(matching, "greedy_match_cells", _fine_matches_stub)
    monkeypatch.setattr(
        matching,
        "_estimate_coarse_transform_from_matches",
        lambda *_args, **_kwargs: (
            candidate_transform,
            np.array([25.0, -10.0], dtype=float),
            "rigid",
            5,
            0.4,
        ),
    )

    result = matching.two_stage_match_cells(
        feats1,
        feats2,
        (400, 400),
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=1.0,
        top_k=10,
        distance_threshold=None,
        spatial_window_size=90.0,
        min_cells_for_two_stage=10,
    )

    assert result.coarse_inlier_count == 5
    assert np.isclose(result.coarse_inlier_ratio, 5.0 / 18.0)
    assert not result.coarse_transform_accepted
    np.testing.assert_allclose(
        result.aligned_df2[["centroid_x", "centroid_y"]].to_numpy(),
        feats2[["centroid_x", "centroid_y"]].to_numpy(),
    )


def test_two_stage_applies_reliable_coarse_transform(monkeypatch):
    feats1 = _make_feature_table()
    feats2 = _make_feature_table()
    candidate_transform = AffineTransform(translation=(25.0, -10.0))

    monkeypatch.setattr(matching, "match_cells_per_patch", _coarse_matches_stub)
    monkeypatch.setattr(matching, "greedy_match_cells", _fine_matches_stub)
    monkeypatch.setattr(
        matching,
        "_estimate_coarse_transform_from_matches",
        lambda *_args, **_kwargs: (
            candidate_transform,
            np.array([25.0, -10.0], dtype=float),
            "rigid",
            7,
            0.4,
        ),
    )

    result = matching.two_stage_match_cells(
        feats1,
        feats2,
        (400, 400),
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=1.0,
        top_k=10,
        distance_threshold=None,
        spatial_window_size=90.0,
        min_cells_for_two_stage=10,
    )

    assert result.coarse_transform_accepted
    np.testing.assert_allclose(
        result.aligned_df2[["centroid_x", "centroid_y"]].to_numpy(),
        candidate_transform(feats2[["centroid_x", "centroid_y"]].to_numpy()),
    )
