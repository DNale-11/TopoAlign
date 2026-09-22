"""Create small label masks for the TopoAlign installation check.

Run from the repository root: python docs/examples/create_demo_masks.py
The moving mask is shifted +6 pixels in x and -4 pixels in y.
These synthetic shapes are an installation check, not a registration benchmark.
"""

from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    output = Path("demo")
    paths = [output / "fixed_mask.tif", output / "moving_mask.tif"]
    if any(path.exists() for path in paths):
        raise SystemExit("Demo mask files already exist; move them before running again.")

    y, x = np.mgrid[:256, :256]
    fixed = np.zeros((256, 256), dtype=np.uint16)
    moving = np.zeros_like(fixed)
    # Each row is (center_x, center_y, radius_x, radius_y).
    cells = [
        (38, 36, 7, 10), (93, 42, 11, 6), (155, 34, 8, 8), (215, 48, 12, 9),
        (48, 108, 9, 12), (110, 100, 6, 8), (173, 110, 10, 7), (224, 119, 7, 11),
        (33, 189, 12, 6), (98, 203, 8, 13), (160, 181, 11, 10), (214, 205, 9, 7),
    ]
    for label, (cx, cy, rx, ry) in enumerate(cells, start=1):
        fixed[((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1] = label
        moving[((x - cx - 6) / rx) ** 2 + ((y - cy + 4) / ry) ** 2 <= 1] = label

    output.mkdir(parents=True, exist_ok=True)
    for path, mask in zip(paths, (fixed, moving)):
        tifffile.imwrite(path, mask, photometric="minisblack")
    print("Created demo/fixed_mask.tif and demo/moving_mask.tif (12 cells each).")
    print("Expected moving-to-fixed translation: x = -6 px, y = +4 px.")


if __name__ == "__main__":
    main()
