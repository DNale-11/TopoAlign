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
    # We will assume this is run in an environment where cellpose is installed.
    # If not, this module will fail on import which is expected.
    # In the sandbox, we might not have it, but we code against the API.
    models = None
    # raise ImportError(
    #     "cellpose is required for segmentation. Install with `pip install cellpose` "
    #     "and ensure Cellpose-SAM weights are available."
    # ) from exc


class CellposeSegmenter:
    """Segment images using Cellpose-SAM."""

    def __init__(self, config: CellposeConfig):
        self.config = config
        if models:
            model_kwargs = {"gpu": config.gpu}
            if config.pretrained_model:
                model_kwargs["pretrained_model"] = config.pretrained_model
            if config.model_type:
                model_kwargs["model_type"] = config.model_type
            self.model = models.CellposeModel(**model_kwargs)
        else:
            self.model = None

    def _select_channel(self, img: np.ndarray) -> np.ndarray:
        """Select a single channel for segmentation (assumes DAPI is last if multiple)."""
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            return img[..., -1]
        raise ValueError(f"Unsupported image shape {img.shape}; expected 2D or 3D.")

    def segment_array(self, img: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
        """
        Segment a numpy array (2D) and return masks along with Cellpose outputs.

        Returns
        -------
        Tuple[np.ndarray, dict, np.ndarray]
            (masks, flows, styles)
        """
        if self.model is None:
            raise RuntimeError("Cellpose model not initialized (missing dependency).")

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

    def segment_zstack(self, img: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
        """
        Segment a 3D Z-stack image.

        Parameters
        ----------
        img : np.ndarray
            Array of shape (Z, Y, X) or (Z, Y, X, C).

        Returns
        -------
        tuple[np.ndarray, dict, np.ndarray]
            (mask_3d, flows, styles)
        """
        if self.model is None:
            raise RuntimeError("Cellpose model not initialized (missing dependency).")

        # Determine if we have a channel dimension
        # The docs say: (nplanes, channels, nY, nX) OR (nplanes, nY, nX)
        # But also CLI/notebook need channel_axis and z_axis parameters.
        # User prompt says: "Multiplane images should be of shape nplanes x channels x nY x nX or as nplanes x nY x nX."
        # And: "For example an image with 2 channels of shape (1024,1024,2,105,1) can be specified with channel_axis=2 and z_axis=3."
        # Here we assume the input img follows standard numpy order (Z, Y, X) or (Z, Y, X, C).

        # We will support (Z, Y, X) and (Z, Y, X, C).
        if img.ndim == 3:
            # (Z, Y, X)
            channel_axis = None
            z_axis = 0
        elif img.ndim == 4:
            # (Z, Y, X, C) - we assume C is last.
            channel_axis = 3
            z_axis = 0
        else:
            raise ValueError(f"Unsupported Z-stack shape {img.shape}; expected (Z, Y, X) or (Z, Y, X, C).")

        eval_kwargs = dict(
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold, # Ignored in 3D according to docs but we pass it
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
            channels=[0, 0], # Grayscale/one channel assumption if not multi-channel model
            do_3D=self.config.do_3D,
            z_axis=z_axis,
            channel_axis=channel_axis,
            stitch_threshold=self.config.stitch_threshold,
        )

        # Flow smoothing for 3D
        if self.config.flow3D_smooth > 0.0:
            eval_kwargs["flow3D_smooth"] = self.config.flow3D_smooth

        if self.config.anisotropy is not None:
            eval_kwargs["anisotropy"] = self.config.anisotropy

        result = self.model.eval(img, **eval_kwargs)

        if len(result) == 4:
            masks_3d, flows, styles, _ = result
        else:
            masks_3d, flows, styles = result

        return masks_3d, flows, styles

    def segment_file(
        self, path: str | Path, save_mask_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """
        Segment an image file and optionally save the resulting mask.
        Detects if 2D or 3D based on image loading.
        """
        img = load_image(path)
        # Simple heuristic: if 3 dimensions and last dim is not small (channels), or 4 dimensions -> Z-stack
        # For now, let's rely on config or shape.
        # Assuming load_image returns standard numpy arrays.
        # We check io_utils.infer_image_mode usually.

        # If infer_image_mode says 3D, we call segment_zstack
        from .io_utils import infer_image_mode
        mode = infer_image_mode(img)

        if mode == "3d_zstack":
            masks, _, _ = self.segment_zstack(img)
        else:
            masks, _, _ = self.segment_array(img)

        if save_mask_path is not None:
            save_mask(save_mask_path, masks)
        return masks
