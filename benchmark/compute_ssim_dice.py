"""
Compute SSIM and DICE for the original synthetic_tps results (pw=1.0).
Reads registered masks from benchmark/results/synthetic_tps/registered_mask/
and compares against target masks from synthetic_dataset/{deform,rigid}/target/
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_ROOT = PROJECT_ROOT / "synthetic_dataset"
RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results" / "synthetic_tps"
MASK_DIR = RESULTS_ROOT / "registered_mask"


def dice_coefficient(mask_a, mask_b):
    """Binary DICE: treat any nonzero pixel as foreground."""
    a = (mask_a > 0).astype(bool)
    b = (mask_b > 0).astype(bool)
    intersection = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_metrics_for_case(case_id):
    """Load registered mask and target, compute SSIM and DICE."""
    reg_path = MASK_DIR / f"{case_id}_registered_mask.tif"
    if not reg_path.exists():
        return None

    # Parse subset and index
    subset = case_id.split("_")[0]  # deform or rigid
    idx = "_".join(case_id.split("_")[1:])
    target_path = SYNTHETIC_ROOT / subset / "target" / f"{idx}_target.tif"

    if not target_path.exists():
        print(f"Warning: target not found for {case_id}")
        return None

    reg_mask = imread(str(reg_path))
    target_img = imread(str(target_path))

    # SSIM on the raw images (target image vs registered mask as image)
    # For instance masks, convert to binary for SSIM
    reg_binary = (reg_mask > 0).astype(np.float64)
    target_binary = (target_img > 0).astype(np.float64)

    # But target is the raw image, not a mask — we need to segment it too
    # Actually, let's also load the target mask by segmenting, or compare
    # registered mask binary against target image binary
    # Wait — target_img is the raw fluorescence image, not a mask.
    # For SSIM we should compare image-to-image.
    # Let's compute:
    # 1. SSIM between target image and the moving image warped (but we don't have warped image)
    # 2. DICE between registered mask (binary) and target mask (binary)
    # We need the target MASK. Let's segment the target to get it.
    # But that's expensive. Instead, use the features CSV which has cell count.
    # Actually the simplest: segment on the fly, or use cached masks.

    # For DICE: we need target mask. Let's segment it.
    return {
        "target_img": target_img,
        "reg_mask": reg_mask,
        "target_path": target_path,
    }


# We need target masks. Let's segment all targets once.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import CellposeConfig, CellposeSegmenter

CELLPOSE_CFG = CellposeConfig(
    gpu=True, pretrained_model="cpsam", diameter=15,
    flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
)
segmenter = CellposeSegmenter(CELLPOSE_CFG)

# Collect all cases
reg_files = sorted(MASK_DIR.glob("*_registered_mask.tif"))
print(f"Found {len(reg_files)} registered masks")

rows = []
seg_cache = {}

for i, reg_path in enumerate(reg_files):
    case_id = reg_path.stem.replace("_registered_mask", "")
    subset = case_id.split("_")[0]
    idx = "_".join(case_id.split("_")[1:])
    target_path = SYNTHETIC_ROOT / subset / "target" / f"{idx}_target.tif"
    moving_path = SYNTHETIC_ROOT / subset / "moving" / f"{idx}_moving.tif"

    if not target_path.exists():
        continue

    # Load registered mask
    reg_mask = imread(str(reg_path))

    # Segment target (cached)
    tk = str(target_path)
    if tk not in seg_cache:
        target_img = imread(str(target_path))
        target_mask, _, _ = segmenter.segment_array(np.asarray(target_img))
        seg_cache[tk] = (target_img, target_mask)
    target_img, target_mask = seg_cache[tk]

    # Load moving image for SSIM (source vs target)
    moving_img = imread(str(moving_path))

    # ── DICE (binary foreground overlap) ──
    dice = dice_coefficient(reg_mask, target_mask)

    # ── SSIM ──
    # Compare target image vs moving image (both uint8 grayscale)
    # After registration, the registered mask aligns moving to target space.
    # For SSIM, we ideally want the warped IMAGE, not just the mask.
    # Since we only saved masks, compute SSIM on binary mask images as proxy.
    reg_binary = (reg_mask > 0).astype(np.uint8) * 255
    target_binary = (target_mask > 0).astype(np.uint8) * 255
    ssim_val = ssim(target_binary, reg_binary, data_range=255)

    if (i + 1) % 20 == 0 or i == 0:
        print(f"[{i+1}/{len(reg_files)}] {case_id}: DICE={dice:.4f}, SSIM={ssim_val:.4f}")

    rows.append({
        "case_id": case_id,
        "subset": subset,
        "dice": dice,
        "ssim": ssim_val,
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS_ROOT / "ssim_dice.csv", index=False)

# Summary
print("\n" + "=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"  Mean DICE: {df['dice'].mean():.4f}")
print(f"  Mean SSIM: {df['ssim'].mean():.4f}")

for subset in ("deform", "rigid"):
    sub = df[df["subset"] == subset]
    if len(sub) > 0:
        print(f"\n  [{subset.upper()}] n={len(sub)}")
        print(f"    DICE: mean={sub['dice'].mean():.4f}  median={sub['dice'].median():.4f}  "
              f"min={sub['dice'].min():.4f}  max={sub['dice'].max():.4f}")
        print(f"    SSIM: mean={sub['ssim'].mean():.4f}  median={sub['ssim'].median():.4f}  "
              f"min={sub['ssim'].min():.4f}  max={sub['ssim'].max():.4f}")
