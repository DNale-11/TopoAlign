"""Feature extraction for segmented cells."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from .config import CellFeaturesConfig

# Core properties needed for matching/registration
BASE_PROPS: List[str] = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "centroid",
]


def _roundness(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    """Compute roundness metric safely."""
    perimeter_safe = np.where(perimeter == 0, np.nan, perimeter)
    roundness = 4 * np.pi * area / (perimeter_safe**2)
    return np.nan_to_num(roundness)


def compute_cell_features(mask: np.ndarray, config: CellFeaturesConfig) -> pd.DataFrame:
    """
    Compute morphological/geometry features for each labeled cell using regionprops.

    Required outputs (for registration/matching):
    - centroid (x, y)
    - area, perimeter
    - roundness, eccentricity, solidity
    - major/minor axis length
    - major axis direction (orientation and unit vector)
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")

    # Collect all properties in a single regionprops_table call (includes area/perimeter/eccentricity/solidity).
    prop_names = list(dict.fromkeys(BASE_PROPS + list(config.extra_properties)))
    props: Dict[str, np.ndarray] = regionprops_table(
        mask,
        properties=prop_names,
    )

    df = pd.DataFrame(props)
    df = df.rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
        }
    )

    df["roundness"] = _roundness(df["area"].to_numpy(), df["perimeter"].to_numpy())
    # Normalized positions in [0,1] relative to image size for spatial matching.
    h, w = mask.shape
    df["pos_x_norm"] = df["centroid_x"] / float(max(w, 1))
    df["pos_y_norm"] = df["centroid_y"] / float(max(h, 1))

    # Orientation (radians) is measured CCW from the horizontal axis to the major axis
    # Provide a unit vector for downstream cosine similarity.
    df["orientation"] = df["orientation"].astype(float)
    df["axis_vec_x"] = np.cos(df["orientation"])
    df["axis_vec_y"] = np.sin(df["orientation"])

    if config.min_area is not None:
        df = df[df["area"] >= config.min_area]
    if config.max_area is not None:
        df = df[df["area"] <= config.max_area]

    df = df.reset_index(drop=True)
    return df
