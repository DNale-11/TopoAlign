"""Quick check: actual displacement for C5-2 pairs."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_tl)
try:
    import torch
except Exception:
    pass

sys.path.insert(0, "napari-cell-registration/src")
sys.path.insert(0, ".")

import numpy as np
from tifffile import imread
from skimage.registration import phase_cross_correlation
from pathlib import Path

root = Path("benchmark/unregistration/C5-2")
fixed = imread(str(root / "fixed" / "C5-C-2.tif"))

for mov_path in sorted((root / "moving").glob("*.tif")):
    moving = imread(str(mov_path))
    # Pad to same size
    h = max(fixed.shape[0], moving.shape[0])
    w = max(fixed.shape[1], moving.shape[1])
    f_pad = np.zeros((h, w), dtype=float)
    m_pad = np.zeros((h, w), dtype=float)
    f_pad[:fixed.shape[0], :fixed.shape[1]] = fixed
    m_pad[:moving.shape[0], :moving.shape[1]] = moving
    shift, _, _ = phase_cross_correlation(f_pad, m_pad, upsample_factor=10)
    mag = np.linalg.norm(shift)
    verdict = "MISS" if mag > 60 else "OK"
    print(f"{mov_path.name}: shift=({shift[0]:.1f}, {shift[1]:.1f})  mag={mag:.1f}px  [{verdict}]")
    print(f"  rematch_window=60px  max_dist=100px")
