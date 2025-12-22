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
    "perimeter", # Only valid for 2D slices or surface area in 3D (requires mesh usually)?
                 # regionprops in 3D does not return 'perimeter' but 'area' is volume.
                 # We will handle this dynamically.
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

    Supports 2D and 3D masks.
    """
    is_3d = mask.ndim == 3

    # Adjust props for 3D
    current_props = list(BASE_PROPS)
    if is_3d:
        # Perimeter is not standard in 3D regionprops (it calculates surface area but via mesh usually or not at all)
        # We'll remove 'perimeter', 'eccentricity', 'orientation' if they cause issues or aren't supported same way.
        # 'area' in 3D regionprops is number of voxels (Volume).
        # 'major_axis_length' etc works in 3D.
        if "perimeter" in current_props:
            current_props.remove("perimeter")
        # 'orientation' in 3D is not a single scalar (it's Euler angles or similar?), regionprops doesn't support 'orientation' for 3D.
        if "orientation" in current_props:
            current_props.remove("orientation")
        # 'eccentricity' is 2D only in skimage regionprops.
        if "eccentricity" in current_props:
            current_props.remove("eccentricity")

    prop_names = list(dict.fromkeys(current_props + list(config.extra_properties)))

    # Filter out properties that might not exist for the dimensionality if manually added
    # regionprops usually raises error if property not supported for dims

    props: Dict[str, np.ndarray] = regionprops_table(
        mask,
        properties=prop_names,
    )

    df = pd.DataFrame(props)

    # Rename centroids
    if is_3d:
        # centroid-0: z, centroid-1: y, centroid-2: x
        df = df.rename(
            columns={
                "label": "cell_id",
                "centroid-0": "centroid_z",
                "centroid-1": "centroid_y",
                "centroid-2": "centroid_x",
            }
        )
        # Fill missing 2D-specific columns with NaN or sensible defaults if needed by downstream
        df["perimeter"] = 0.0
        df["eccentricity"] = 0.0
        df["orientation"] = 0.0

        # We can add 'volume' alias for area
        df["volume"] = df["area"]

    else:
        df = df.rename(
            columns={
                "label": "cell_id",
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
            }
        )
        # Calculate 2D specific metrics
        df["roundness"] = _roundness(df["area"].to_numpy(), df["perimeter"].to_numpy())

        # Orientation
        df["orientation"] = df["orientation"].astype(float)
        df["axis_vec_x"] = np.cos(df["orientation"])
        df["axis_vec_y"] = np.sin(df["orientation"])

    # Normalized positions in [0,1]
    if is_3d:
        d, h, w = mask.shape
        df["pos_z_norm"] = df["centroid_z"] / float(max(d, 1))
        df["pos_y_norm"] = df["centroid_y"] / float(max(h, 1))
        df["pos_x_norm"] = df["centroid_x"] / float(max(w, 1))
    else:
        h, w = mask.shape
        df["pos_y_norm"] = df["centroid_y"] / float(max(h, 1))
        df["pos_x_norm"] = df["centroid_x"] / float(max(w, 1))

    if config.min_area is not None:
        df = df[df["area"] >= config.min_area]
    if config.max_area is not None:
        df = df[df["area"] <= config.max_area]

    df = df.reset_index(drop=True)
    return df
