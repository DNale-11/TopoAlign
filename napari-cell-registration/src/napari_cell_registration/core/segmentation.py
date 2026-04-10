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

    def _select_channel_zstack(self, img: np.ndarray) -> np.ndarray:
        """Select the DAPI channel for Z-stacks (Z, Y, X[, C])."""
        if img.ndim == 3:
            return img  # single-channel Z-stack
        if img.ndim == 4:
            return img[..., -1]  # assume DAPI is last channel
        raise ValueError(f"Unsupported Z-stack shape {img.shape}; expected (Z, Y, X) or (Z, Y, X, C).")

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

    def segment_zstack(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        """
        Segment a 3D Z-stack image and return both 3D and projected 2D masks.

        Parameters
        ----------
        img : np.ndarray
            Array of shape (Z, Y, X) or (Z, Y, X, C).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict, np.ndarray]
            (mask_3d, mask_2d, flows, styles) where mask_2d is a max projection of mask_3d.
        """
        channel_img = self._select_channel_zstack(img)
        eval_kwargs = dict(
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
            channels=[0, 0],
            do_3D=True,
            # Explicit axes for 3D: z is axis 0, channel_axis None for ZYX or last for ZYXC
            z_axis=0,
        )
        if channel_img.ndim == 4:
            eval_kwargs["channel_axis"] = -1
        else:
            eval_kwargs["channel_axis"] = None
        if self.config.anisotropy is not None:
            eval_kwargs["anisotropy"] = self.config.anisotropy
        result = self.model.eval(channel_img, **eval_kwargs)
        if len(result) == 4:
            masks_3d, flows, styles, _ = result
        else:
            masks_3d, flows, styles = result
        masks_2d = project_labels_max(masks_3d)
        return masks_3d, masks_2d, flows, styles

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


def project_labels_max(mask_3d: np.ndarray) -> np.ndarray:
    """
    Project a 3D label volume (Z, Y, X) into a 2D label image (Y, X) using max across Z.

    If multiple labels overlap along Z at the same (Y, X), the highest label id is kept.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected a 3D mask to project, got shape {mask_3d.shape}.")
    return np.max(mask_3d, axis=0)
