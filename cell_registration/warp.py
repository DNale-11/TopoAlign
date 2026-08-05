"""Image and mask warping helpers for TopoAlign CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.transform import AffineTransform, warp


ChannelAxis = Literal["auto", "first", "last", "none"]


def load_transform(path: str | Path) -> AffineTransform:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = payload.get("matrix") if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and matrix is None:
        for key in ("moving_to_fixed", "matrix", "affine", "transform"):
            if key in payload:
                matrix = payload[key]
                break
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Transform must contain a 3x3 matrix, got {matrix.shape}.")
    return AffineTransform(matrix=matrix)


def _cast_like(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
    return arr.astype(dtype, copy=False)


def _warp_plane(plane: np.ndarray, transform: AffineTransform, output_shape: tuple[int, int], order: int) -> np.ndarray:
    return warp(
        plane.astype(float),
        inverse_map=transform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        order=order,
        mode="constant",
        cval=0.0,
    )


def warp_array(
    moving: np.ndarray,
    transform: AffineTransform,
    output_shape: tuple[int, int],
    *,
    channel_axis: ChannelAxis = "auto",
    order: int = 1,
) -> np.ndarray:
    """Warp moving data into fixed coordinates using moving->fixed transform."""
    arr = np.asarray(moving)
    dtype = arr.dtype
    if arr.ndim == 2 or channel_axis == "none":
        if arr.ndim != 2:
            raise ValueError("channel_axis='none' requires a 2D array.")
        return _cast_like(_warp_plane(arr, transform, output_shape, order), dtype)
    if arr.ndim != 3:
        raise ValueError(f"Only 2D, CYX, or YXC arrays are supported; got {arr.shape}.")
    if channel_axis == "first":
        return _cast_like(np.stack([_warp_plane(arr[c], transform, output_shape, order) for c in range(arr.shape[0])]), dtype)
    if channel_axis == "last":
        return _cast_like(np.stack([_warp_plane(arr[..., c], transform, output_shape, order) for c in range(arr.shape[-1])], axis=-1), dtype)
    if arr.shape[-1] <= 4:
        return _cast_like(np.stack([_warp_plane(arr[..., c], transform, output_shape, order) for c in range(arr.shape[-1])], axis=-1), dtype)
    # Auto mode treats a 3D array with many planes as a Z,Y,X stack.
    return _cast_like(np.stack([_warp_plane(arr[c], transform, output_shape, order) for c in range(arr.shape[0])]), dtype)


def valid_overlap_mask(source_shape: tuple[int, int], transform: AffineTransform, output_shape: tuple[int, int]) -> np.ndarray:
    ones = np.ones(source_shape, dtype=np.uint8)
    return _warp_plane(ones, transform, output_shape, order=0) > 0.5
