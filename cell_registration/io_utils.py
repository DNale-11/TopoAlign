"""Image I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np

from .config import SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array as float32. Channels are preserved if present.
    """
    img = iio.imread(path)
    return np.asarray(img, dtype=np.float32)


def find_images(directory: str | Path) -> List[Path]:
    """
    Recursively find supported image files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search.

    Returns
    -------
    List[Path]
        List of file paths.
    """
    root = Path(directory)
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """
    Save a label mask to disk.

    Parameters
    ----------
    path : str or Path
        Destination path.
    mask : np.ndarray
        Label mask to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, mask.astype(np.int32))
