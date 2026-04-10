"""Core functionality for cell registration."""

from .config import (
    CellposeConfig,
    CellFeaturesConfig,
    MatchingConfig,
    DEFAULT_CELLPOSE_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_MATCHING_CONFIG,
)
from .segmentation import CellposeSegmenter
from .features import compute_cell_features
from .matching import greedy_match_cells, match_cells_per_patch, match_cells_per_cluster
from .registration import estimate_rigid_transform_from_matches
from .validation import validate_matches, compute_match_quality_stats

__all__ = [
    "CellposeConfig",
    "CellFeaturesConfig",
    "MatchingConfig",
    "DEFAULT_CELLPOSE_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_MATCHING_CONFIG",
    "CellposeSegmenter",
    "compute_cell_features",
    "greedy_match_cells",
    "match_cells_per_patch",
    "match_cells_per_cluster",
    "estimate_rigid_transform_from_matches",

    "validate_matches",
    "compute_match_quality_stats",
]
