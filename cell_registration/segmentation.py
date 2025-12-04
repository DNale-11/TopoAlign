"""Cellpose-SAM segmentation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import CellposeConfig
from .io_utils import load_image, save_mask

try:
    from cellpose import models
except ImportError as exc:
    raise ImportError(
        "cellpose is required for segmentation. Install with `pip install cellpose` "
        "and ensure Cellpose-SAM weights are available."
    ) from exc


class CellposeSegmenter:
    """Segment images using Cellpose-SAM."""

    def __init__(self, config: CellposeConfig):
        self.config = config
        model_kwargs = {"gpu": config.gpu}
        # Prefer explicit built-in/custom checkpoint name; model_type is only for older cellpose.
        if config.pretrained_model:
            model_kwargs["pretrained_model"] = config.pretrained_model
        if config.model_type:
            model_kwargs["model_type"] = config.model_type
        self.model = models.CellposeModel(**model_kwargs)

    def _select_channel(self, img: np.ndarray) -> np.ndarray:
        """Select a single channel for segmentation (assumes DAPI is last if multiple)."""
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            return img[..., -1]
        raise ValueError(f"Unsupported image shape {img.shape}; expected 2D or 3D.")

    def segment_array(self, img: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
        """
        Segment a numpy array and return masks along with Cellpose outputs.

        Returns
        -------
        Tuple[np.ndarray, dict, np.ndarray]
            (masks, flows, styles)
        """
        channel_img = self._select_channel(img)
        # cellpose 4.x returns (masks, flows, styles); older versions returned 4 items.
        result = self.model.eval(
            channel_img,
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
            channels=[0, 0],
        )
        if len(result) == 4:
            masks, flows, styles, _ = result
        else:
            masks, flows, styles = result
        return masks, flows, styles

    def segment_file(
        self, path: str | Path, save_mask_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """
        Segment an image file and optionally save the resulting mask.

        Parameters
        ----------
        path : str or Path
            Image path.
        save_mask_path : str or Path, optional
            If provided, save the mask to this path.

        Returns
        -------
        np.ndarray
            Label mask.
        """
        img = load_image(path)
        masks, _, _ = self.segment_array(img)
        if save_mask_path is not None:
            save_mask(save_mask_path, masks)
        return masks
