from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NAPARI_SRC = PROJECT_ROOT / "napari-cell-registration" / "src"
if str(NAPARI_SRC) not in sys.path:
    sys.path.insert(0, str(NAPARI_SRC))


def _configure_torch_runtime_path() -> None:
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if not os.path.isdir(torch_lib):
        torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)


_configure_torch_runtime_path()

from cell_registration.evaluation import compute_object_registration_metrics
from registration_runtime import (
    DEFAULT_REG_PARAMS,
    build_segmenter,
    load_segmentation_artifacts,
    run_registration,
)
from run_unregistration_batch import DEFAULT_INPUT_ROOT, discover_unregistration_cases
import pandas as pd


PROFILE_GRID = [
    ("pw3_dt2.5_win100", 3.0, 2.5, 100.0),
    ("pw8_dt2_win80", 8.0, 2.0, 80.0),
    ("pw12_dt1.8_win80", 12.0, 1.8, 80.0),
    ("pw15_dt1.5_win70", 15.0, 1.5, 70.0),
    ("pw20_dt1_win60", 20.0, 1.0, 60.0),
    ("pw20_dt1.5_win70", 20.0, 1.5, 70.0),
    ("pw20_dt2_win80", 20.0, 2.0, 80.0),
]


def _parse_case_ids(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _build_profile_params(position_weight: float, distance_threshold: float, spatial_window_size: float) -> dict:
    params = dict(DEFAULT_REG_PARAMS)
    params.update(
        {
            "feature_weight": 1.0,
            "position_weight": position_weight,
            "distance_threshold": distance_threshold,
            "spatial_window_size": spatial_window_size,
            "coarse_distance_threshold": min(2.5, distance_threshold + 0.8),
            "guided_min_position_weight": max(8.0, position_weight),
            "guided_spatial_window_cap": min(spatial_window_size, max(40.0, spatial_window_size - 20.0)),
            "validation_max_feature_diff": 0.7,
            "gpu_enabled": True,
        }
    )
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a small profile grid on selected unregistration cases.")
    parser.add_argument(
        "--case-ids",
        required=True,
        help="Comma-separated case ids, e.g. A2-3,A4-1,C3-1",
    )
    args = parser.parse_args()

    case_ids = _parse_case_ids(args.case_ids)
    cases, failures = discover_unregistration_cases(DEFAULT_INPUT_ROOT, selected_case_ids=case_ids)
    if failures:
        raise RuntimeError(f"Discovery failures: {failures}")

    segmenter = build_segmenter(dict(DEFAULT_REG_PARAMS), gpu=True)
    cache = {}
    rows: list[dict] = []

    for case in cases:
        fixed_artifacts, _ = load_segmentation_artifacts(case.fixed_path, segmenter, cache)
        for moving_path in case.moving_paths:
            moving_artifacts, _ = load_segmentation_artifacts(moving_path, segmenter, cache)
            for profile_name, pw, dt, win in PROFILE_GRID:
                params = _build_profile_params(pw, dt, win)
                registration = run_registration(moving_artifacts, fixed_artifacts, params, logger=lambda _msg: None)
                metrics = compute_object_registration_metrics(
                    moving_mask=registration.moving_mask,
                    fixed_mask=registration.fixed_mask,
                    moving_features=registration.moving_features,
                    fixed_features=registration.fixed_features,
                    transform=registration.transform,
                    valid_mask=registration.valid_mask,
                    instance_iou_threshold=0.3,
                    min_valid_instance_area_px=20,
                    min_valid_fraction=0.5,
                )
                rows.append(
                    {
                        "profile": profile_name,
                        "case_id": case.case_id,
                        "moving_name": moving_path.name,
                        "match_f1": metrics["match_f1"],
                        "mask_dice": metrics["mask_dice"],
                        "centroid_error_median_px": metrics["centroid_error_median_px"],
                        "inlier_count": registration.diagnostics.get("inlier_count", 0),
                        "median_inlier_residual": registration.diagnostics.get("median_inlier_residual", float("inf")),
                    }
                )

    df = pd.DataFrame(rows)
    summary_rows = []
    for profile, group in df.groupby("profile"):
        worst = group.nsmallest(min(8, len(group)), "match_f1")
        summary_rows.append(
            {
                "profile": profile,
                "mean_f1": group["match_f1"].mean(),
                "mean_dice": group["mask_dice"].mean(),
                "mean_tre": group["centroid_error_median_px"].mean(),
                "worst8_f1": worst["match_f1"].mean(),
                "worst8_dice": worst["mask_dice"].mean(),
                "worst8_tre": worst["centroid_error_median_px"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["worst8_f1", "mean_f1"], ascending=[False, False])
    print(summary_df.to_string(index=False))
    print("\nBest profile per pair:")
    idx = df.groupby(["case_id", "moving_name"])["match_f1"].idxmax()
    best = df.loc[idx].sort_values("match_f1")
    print(
        best[
            [
                "case_id",
                "moving_name",
                "profile",
                "match_f1",
                "mask_dice",
                "centroid_error_median_px",
                "inlier_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
