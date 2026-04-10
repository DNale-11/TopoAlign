from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cell_registration.main as main


def test_rebuild_matches_from_topology_pairs_enforces_one_to_one(monkeypatch):
    fixed_feats = pd.DataFrame(
        {
            "cell_id": [101, 102],
            "centroid_x": [10.0, 30.0],
            "centroid_y": [10.0, 30.0],
        }
    )
    moving_feats = pd.DataFrame(
        {
            "cell_id": [201, 202],
            "centroid_x": [12.0, 32.0],
            "centroid_y": [9.0, 31.0],
        }
    )
    base_matches = pd.DataFrame(
        {
            "idx1": [0, 1],
            "idx2": [0, 1],
            "distance": [0.10, 0.20],
        }
    )

    trusted_pairs = pd.DataFrame(
        {
            "cell_id_r1": [101, 102],
            "cell_id_r2": [201, 202],
            "L_pos": [0.10, 0.20],
            "L_nei": [0.05, 0.05],
            "patch_x": [0, 0],
            "patch_y": [0, 0],
        }
    )
    neighbor_pairs = pd.DataFrame(
        {
            "cell_id_r1": [101],
            "cell_id_r2": [202],
            "dist_norm": [0.01],
            "within_threshold": [True],
            "patch_x": [0],
            "patch_y": [0],
        }
    )

    monkeypatch.setattr(
        main,
        "run_topology_matching_df",
        lambda *args, **kwargs: (trusted_pairs, neighbor_pairs),
    )

    rebuilt = main.rebuild_matches_from_topology_pairs(
        base_matches,
        fixed_feats,
        moving_feats,
        image_width=64.0,
        image_height=64.0,
        k_pos_nei=5,
        k_neighbor=5,
        tau_pos=0.5,
        tau_nei=0.3,
        tau_map=0.3,
    )

    assert list(rebuilt["cell_id_1"]) == [101, 102]
    assert list(rebuilt["cell_id_2"]) == [201, 202]
    assert list(rebuilt["distance"]) == [0.10, 0.20]
