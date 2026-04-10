"""Global configuration objects and defaults for cell registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class CellposeConfig:
    """Configuration for Cellpose-SAM segmentation."""

    gpu: bool = True
    # Use SAM checkpoint by default.
    pretrained_model: str = "cpsam"
    # model_type is ignored in cellpose 4.x; keep None.
    model_type: Optional[str] = None
    diameter: Optional[float] = 8
    flow_threshold: float = -2
    cellprob_threshold: float = -2
    min_size: int = 1
    # Optional anisotropy for 3D Z-stacks (None -> isotropic / default behavior).
    anisotropy: Optional[float] = None


@dataclass
class CellFeaturesConfig:
    """Configuration for cell feature extraction."""

    min_area: Optional[int] = None
    max_area: Optional[int] = None
    extra_properties: Tuple[str, ...] = field(default_factory=tuple)


# Default instances that can be imported elsewhere
DEFAULT_CELLPOSE_CONFIG = CellposeConfig()
DEFAULT_FEATURE_CONFIG = CellFeaturesConfig()
