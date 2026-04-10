"""Cell registration package: segmentation, feature extraction, matching, and rigid alignment."""

from .config import CellposeConfig, CellFeaturesConfig, DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
try:
    from .segmentation import CellposeSegmenter
except ImportError:
    # Allow package import without cellpose if segmentation is not used
    CellposeSegmenter = None
from .features import compute_cell_features
from .matching import MatchingConfig, greedy_match_cells
from .registration import RigidTransform, estimate_rigid_transform_from_matches
from .robust_alignment import perform_global_registration
from .visualization import launch_napari_viewer, export_match_table, save_match_overlay
from .mask_matching import MaskMatchConfig, MaskMatchResult, match_cells_by_mask_overlap, match_masks_pipeline

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
    "perform_global_registration",
    "launch_napari_viewer",
    "export_match_table",
    "save_match_overlay",
    "MaskMatchConfig",
    "MaskMatchResult",
    "match_cells_by_mask_overlap",
    "match_masks_pipeline",
]

