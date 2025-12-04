"""Cell registration package: segmentation, feature extraction, matching, and rigid alignment."""

from .config import CellposeConfig, CellFeaturesConfig, DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .segmentation import CellposeSegmenter
from .features import compute_cell_features
from .matching import MatchingConfig, greedy_match_cells
from .registration import RigidTransform, estimate_rigid_transform_from_matches
from .visualization import launch_napari_viewer, export_match_table, save_match_overlay

__all__ = [
    "CellposeConfig",
    "CellFeaturesConfig",
    "DEFAULT_CELLPOSE_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "CellposeSegmenter",
    "compute_cell_features",
    "MatchingConfig",
    "greedy_match_cells",
    "RigidTransform",
    "estimate_rigid_transform_from_matches",
    "launch_napari_viewer",
    "export_match_table",
    "save_match_overlay",
]
