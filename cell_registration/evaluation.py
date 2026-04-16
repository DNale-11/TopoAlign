import numpy as np
import pandas as pd
from skimage.transform import warp, AffineTransform

def compute_overlap_metrics(reg, ref, valid_mask=None):
    return {}

def compute_object_registration_metrics(
    moving_mask, fixed_mask, moving_features, fixed_features, transform, valid_mask=None,
    instance_iou_threshold=0.3, min_valid_instance_area_px=20, min_valid_fraction=0.5,
    return_match_table=False,
):
    if hasattr(transform, 'inverse'):
        inverse_map = transform.inverse
    else:
        inverse_map = AffineTransform(matrix=np.linalg.inv(np.asarray(transform.params))).inverse
        
    warped_moving = np.rint(warp(
        moving_mask.astype(float),
        inverse_map=inverse_map,
        output_shape=fixed_mask.shape[:2],
        preserve_range=True,
        order=0
    )).astype(int)

    if valid_mask is not None:
        warped_moving[~valid_mask] = 0

    cells_m = np.unique(warped_moving)[1:]
    cells_f = np.unique(fixed_mask)[1:]

    eligible_moving = len(cells_m)
    eligible_fixed = len(cells_f)

    intersect = np.stack([warped_moving, fixed_mask], axis=-1)
    intersect = intersect[(warped_moving > 0) & (fixed_mask > 0)]
    
    unique, counts = np.unique(intersect, axis=0, return_counts=True)
    
    area_m = {c: int((warped_moving == c).sum()) for c in cells_m}
    area_f = {c: int((fixed_mask == c).sum()) for c in cells_f}

    tp = 0
    ious = []
    
    for (m, f), count in zip(unique, counts):
        if m in area_m and f in area_f:
            iou = count / float(area_m[m] + area_f[f] - count)
            if iou >= instance_iou_threshold:
                tp += 1
                ious.append(iou)

    fp = eligible_moving - tp
    fn = eligible_fixed - tp

    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)

    metrics = {
        "match_f1": f1,
        "match_precision": precision,
        "match_recall": recall,
        "matched_cells": tp,
        "eligible_moving_cells": eligible_moving,
        "eligible_fixed_cells": eligible_fixed,
        "matched_mean_iou": float(np.mean(ious)) if ious else float('nan'),
        "tre_mean_px": float('nan'),
        "tre_median_px": float('nan'),
        "tre_p95_px": float('nan'),
        "mask_dice": float('nan'),
        "mask_iou": float('nan')
    }
    
    if return_match_table:
        return metrics, pd.DataFrame()
    return metrics


def compute_prewarped_registration_metrics(
    warped_moving_mask,
    fixed_mask,
    *,
    instance_iou_threshold=0.3,
    return_match_table=False,
):
    """Evaluate F1 using an already-warped moving mask (e.g. from TPS warp)."""
    warped_moving = np.asarray(warped_moving_mask, dtype=int)
    fixed = np.asarray(fixed_mask, dtype=int)

    cells_m = np.unique(warped_moving)[1:]
    cells_f = np.unique(fixed)[1:]

    eligible_moving = len(cells_m)
    eligible_fixed = len(cells_f)

    intersect = np.stack([warped_moving, fixed], axis=-1)
    intersect = intersect[(warped_moving > 0) & (fixed > 0)]

    unique, counts = np.unique(intersect, axis=0, return_counts=True)

    area_m = {c: int((warped_moving == c).sum()) for c in cells_m}
    area_f = {c: int((fixed == c).sum()) for c in cells_f}

    tp = 0
    ious = []

    for (m, f), count in zip(unique, counts):
        if m in area_m and f in area_f:
            iou = count / float(area_m[m] + area_f[f] - count)
            if iou >= instance_iou_threshold:
                tp += 1
                ious.append(iou)

    fp = eligible_moving - tp
    fn = eligible_fixed - tp

    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)

    metrics = {
        "match_f1": f1,
        "match_precision": precision,
        "match_recall": recall,
        "matched_cells": tp,
        "eligible_moving_cells": eligible_moving,
        "eligible_fixed_cells": eligible_fixed,
        "matched_mean_iou": float(np.mean(ious)) if ious else float('nan'),
    }

    if return_match_table:
        return metrics, pd.DataFrame()
    return metrics
