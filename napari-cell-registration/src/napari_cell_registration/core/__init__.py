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
from .matching import greedy_match_cells, match_cells_per_cluster, two_stage_match_cells
from .registration import estimate_rigid_transform_from_matches
from .validation import validate_matches, compute_match_quality_stats
from .robust_alignment import perform_global_registration, apply_transform_to_coordinates
from .workflow import (
    MAIN_MATCHING_FEATURE_COLUMNS,
    MIN_MATCHES_FOR_REFINEMENT,
    assign_patches,
    assign_clusters_from_round1,
    run_topology_matching_df,
    compute_match_residuals,
    estimate_rigid_transform_from_matches_ransac,
    rigid_transform_to_affine,
    apply_rigid_to_points,
    cast_warped_like_original,
)

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
    "match_cells_per_cluster",
    "two_stage_match_cells",
    "estimate_rigid_transform_from_matches",
    "perform_global_registration",
    "apply_transform_to_coordinates",
    "MAIN_MATCHING_FEATURE_COLUMNS",
    "MIN_MATCHES_FOR_REFINEMENT",
    "assign_patches",
    "assign_clusters_from_round1",
    "run_topology_matching_df",
    "compute_match_residuals",
    "estimate_rigid_transform_from_matches_ransac",
    "rigid_transform_to_affine",
    "apply_rigid_to_points",
    "cast_warped_like_original",

    "validate_matches",
    "compute_match_quality_stats",
]
