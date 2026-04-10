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
                        "orientation": 0.0,
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


def _make_feature_row(
    cell_id: int,
    x: float,
    y: float,
    *,
    area: float = 100.0,
    perimeter: float = 40.0,
    roundness: float = 0.8,
    eccentricity: float = 0.7,
    solidity: float = 0.9,
    major_axis_length: float = 12.0,
    minor_axis_length: float = 8.0,
    orientation: float = 0.0,
) -> dict:
    return {
        "cell_id": cell_id,
        "centroid_x": x,
        "centroid_y": y,
        "pos_x_norm": x / 400.0,
        "pos_y_norm": y / 400.0,
        "area": area,
        "perimeter": perimeter,
        "roundness": roundness,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "aspect_ratio": major_axis_length / minor_axis_length,
        "elongation": 1.0 - (minor_axis_length / major_axis_length),
        "equivalent_diameter": np.sqrt(4.0 * area / np.pi),
        "orientation": orientation,
        "nn_dist_1": 1.0,
        "nn_dist_2": 1.0,
        "nn_dist_3": 1.0,
        "local_density": 1.0,
        "neighbor_area_ratio_mean": 1.0,
        "neighbor_roundness_mean": roundness,
    }


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


def test_local_morphology_candidates_respect_window_and_hard_constraints():
    fixed = pd.DataFrame(
        [
            _make_feature_row(1, 100.0, 100.0, area=100.0, orientation=0.0),
        ]
    )
    moving = pd.DataFrame(
        [
            _make_feature_row(10, 110.0, 104.0, area=260.0, orientation=0.0),
            _make_feature_row(11, 112.0, 102.0, area=100.0, orientation=np.pi / 2.0),
            _make_feature_row(12, 113.0, 103.0, area=102.0, orientation=0.08),
            _make_feature_row(13, 300.0, 300.0, area=100.0, orientation=0.0),
        ]
    )
    config = matching.MatchingConfig(distance_threshold=None)

    candidates = matching._build_local_morphology_candidates(
        fixed,
        moving,
        config,
        image_shape=(400, 400),
        top_k_per_cell=4,
        spatial_window_size=40.0,
        max_candidates=None,
    )
    filtered = matching._filter_candidate_matches_by_hard_constraints(
        candidates,
        max_area_ratio=2.0,
        max_aspect_ratio_ratio=2.0,
        max_orientation_diff_deg=30.0,
        min_orientation_eccentricity=0.35,
    )

    assert 13 not in set(candidates["cell_id_2"])
    assert list(filtered["cell_id_2"]) == [12]


def test_two_stage_morphology_guided_recovers_translation():
    translation = np.array([18.0, -12.0], dtype=float)
    fixed_rows = []
    moving_rows = []
    base_coords = [
        (70.0, 70.0),
        (150.0, 75.0),
        (240.0, 90.0),
        (85.0, 180.0),
        (165.0, 200.0),
        (255.0, 215.0),
    ]
    for cell_id, (x, y) in enumerate(base_coords, start=1):
        fixed_rows.append(
            _make_feature_row(
                cell_id,
                x,
                y,
                area=100.0 + cell_id,
                perimeter=40.0 + cell_id,
                orientation=0.03 * cell_id,
            )
        )
        moving_rows.append(
            _make_feature_row(
                100 + cell_id,
                x - translation[0],
                y - translation[1],
                area=100.0 + cell_id,
                perimeter=40.0 + cell_id,
                orientation=0.03 * cell_id,
            )
        )

    result = matching.two_stage_match_cells(
        pd.DataFrame(fixed_rows),
        pd.DataFrame(moving_rows),
        (400, 400),
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=1.0,
        top_k=6,
        distance_threshold=None,
        spatial_window_size=25.0,
        min_cells_for_two_stage=4,
        coarse_top_k=18,
        coarse_distance_threshold=3.0,
        coarse_matching_mode="morphology_guided",
        coarse_candidates_per_cell=3,
        coarse_residual_threshold=3.0,
        coarse_min_inlier_count=3,
        coarse_min_inlier_ratio=0.5,
    )

    assert result.coarse_transform_accepted
    np.testing.assert_allclose(result.coarse_offset_xy, translation, atol=1.0)
    assert list(result.matches["cell_id_1"]) == [1, 2, 3, 4, 5, 6]
    assert list(result.matches["cell_id_2"]) == [101, 102, 103, 104, 105, 106]
