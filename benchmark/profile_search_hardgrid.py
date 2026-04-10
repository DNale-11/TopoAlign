from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NAPARI_SRC = PROJECT_ROOT / "napari-cell-registration" / "src"
if str(NAPARI_SRC) not in sys.path:
    sys.path.insert(0, str(NAPARI_SRC))

from registration_runtime import DEFAULT_REG_PARAMS, build_segmenter, load_segmentation_artifacts, run_registration
from run_unregistration_batch import DEFAULT_INPUT_ROOT, discover_unregistration_cases
from cell_registration.evaluation import compute_object_registration_metrics
import pandas as pd


DEFAULT_FEATURE_WEIGHTS = "1.0,0.9,0.85"
DEFAULT_TOP_KS = "100,80"
DEFAULT_GUIDED_TOP_KS = "default,80"
DEFAULT_GUIDED_RELAXATIONS = "0.5,0.0"
DEFAULT_CLIP_MADS = "0.0,2.5"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "profile_search_hardgrid_v1"


def _parse_case_id_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_or_default_list(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"default", "none", "null"}:
            parsed.append(None)
        else:
            parsed.append(int(token))
    return parsed


def _profile_name(
    *,
    feature_weight: float,
    top_k: int,
    guided_top_k: int | None,
    guided_distance_relaxation: float,
    guided_residual_clip_mad_factor: float,
) -> str:
    guided_top_k_name = 100 if guided_top_k is None else int(guided_top_k)
    return (
        f"fw{int(round(feature_weight * 100)):03d}"
        f"_tk{int(top_k)}"
        f"_gtk{guided_top_k_name}"
        f"_rel{int(round(guided_distance_relaxation * 10)):02d}"
        f"_clip{int(round(guided_residual_clip_mad_factor * 10)):02d}"
    )


def _iter_profiles(
    *,
    feature_weights: Iterable[float],
    top_ks: Iterable[int],
    guided_top_ks: Iterable[int | None],
    guided_relaxations: Iterable[float],
    clip_mads: Iterable[float],
) -> list[tuple[str, dict]]:
    profiles: list[tuple[str, dict]] = []
    for feature_weight, top_k, guided_top_k, guided_distance_relaxation, clip_mad in itertools.product(
        feature_weights,
        top_ks,
        guided_top_ks,
        guided_relaxations,
        clip_mads,
    ):
        params = dict(DEFAULT_REG_PARAMS)
        params.update(
            {
                "feature_weight": float(feature_weight),
                "position_weight": 20.0,
                "distance_threshold": 1.0,
                "spatial_window_size": 60.0,
                "top_k": int(top_k),
                "coarse_distance_threshold": 1.8,
                "guided_min_position_weight": 8.0,
                "guided_spatial_window_cap": 40.0,
                "guided_top_k": guided_top_k,
                "guided_distance_relaxation": float(guided_distance_relaxation),
                "guided_residual_clip_mad_factor": float(clip_mad),
                "guided_residual_clip_max_drop_fraction": 0.15,
                "guided_residual_clip_min_median_gain_px": 0.1,
                "gpu_enabled": True,
            }
        )
        profiles.append(
            (
                _profile_name(
                    feature_weight=float(feature_weight),
                    top_k=int(top_k),
                    guided_top_k=guided_top_k,
                    guided_distance_relaxation=float(guided_distance_relaxation),
                    guided_residual_clip_mad_factor=float(clip_mad),
                ),
                params,
            )
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a focused hard-case profile grid around the current no-FFT baseline.")
    parser.add_argument("--case-ids", required=True, help="Comma-separated case ids, e.g. A2-3,A4-1,C3-1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-weights", default=DEFAULT_FEATURE_WEIGHTS)
    parser.add_argument("--top-ks", default=DEFAULT_TOP_KS)
    parser.add_argument("--guided-top-ks", default=DEFAULT_GUIDED_TOP_KS)
    parser.add_argument("--guided-relaxations", default=DEFAULT_GUIDED_RELAXATIONS)
    parser.add_argument("--guided-clip-mads", default=DEFAULT_CLIP_MADS)
    args = parser.parse_args()

    case_ids = _parse_case_id_list(args.case_ids)
    feature_weights = _parse_float_list(args.feature_weights)
    top_ks = [int(value) for value in _parse_float_list(args.top_ks)]
    guided_top_ks = _parse_int_or_default_list(args.guided_top_ks)
    guided_relaxations = _parse_float_list(args.guided_relaxations)
    clip_mads = _parse_float_list(args.guided_clip_mads)

    profiles = _iter_profiles(
        feature_weights=feature_weights,
        top_ks=top_ks,
        guided_top_ks=guided_top_ks,
        guided_relaxations=guided_relaxations,
        clip_mads=clip_mads,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "search_config.json").write_text(
        json.dumps(
            {
                "case_ids": sorted(case_ids),
                "feature_weights": feature_weights,
                "top_ks": top_ks,
                "guided_top_ks": guided_top_ks,
                "guided_relaxations": guided_relaxations,
                "guided_clip_mads": clip_mads,
                "profile_count": len(profiles),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cases, failures = discover_unregistration_cases(DEFAULT_INPUT_ROOT, selected_case_ids=case_ids)
    if failures:
        raise RuntimeError(f"Discovery failures: {failures}")

    segmenter = build_segmenter(dict(DEFAULT_REG_PARAMS), gpu=True)
    cache = {}
    rows: list[dict] = []

    pair_total = sum(len(case.moving_paths) for case in cases)
    print(f"Profiles: {len(profiles)} | Pairs: {pair_total}")

    for case in cases:
        fixed_artifacts, _ = load_segmentation_artifacts(case.fixed_path, segmenter, cache)
        for moving_path in case.moving_paths:
            moving_artifacts, _ = load_segmentation_artifacts(moving_path, segmenter, cache)
            for profile_name, reg_params in profiles:
                registration = run_registration(
                    moving_artifacts,
                    fixed_artifacts,
                    reg_params,
                    logger=lambda _msg: None,
                )
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
                        "fixed_name": case.fixed_path.name,
                        "moving_name": moving_path.name,
                        "match_f1": metrics["match_f1"],
                        "matched_mean_iou": metrics["matched_mean_iou"],
                        "mask_dice": metrics["mask_dice"],
                        "centroid_error_median_px": metrics["centroid_error_median_px"],
                        "centroid_error_p95_px": metrics["centroid_error_p95_px"],
                        "inlier_count": registration.diagnostics.get("inlier_count", 0),
                        "median_inlier_residual": registration.diagnostics.get("median_inlier_residual", float("inf")),
                    }
                )
            print(f"done {case.case_id} / {moving_path.name}")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "detailed_results.csv", index=False)

    summary_rows: list[dict] = []
    for profile, group in results.groupby("profile"):
        worst = group.nsmallest(min(8, len(group)), "match_f1")
        summary_rows.append(
            {
                "profile": profile,
                "mean_f1": group["match_f1"].mean(),
                "mean_iou": group["matched_mean_iou"].mean(),
                "mean_dice": group["mask_dice"].mean(),
                "mean_med": group["centroid_error_median_px"].mean(),
                "mean_p95": group["centroid_error_p95_px"].mean(),
                "mean_inliers": group["inlier_count"].mean(),
                "worst8_f1": worst["match_f1"].mean(),
                "worst8_iou": worst["matched_mean_iou"].mean(),
                "worst8_dice": worst["mask_dice"].mean(),
                "worst8_med": worst["centroid_error_median_px"].mean(),
                "worst8_p95": worst["centroid_error_p95_px"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["worst8_f1", "mean_f1", "mean_iou", "mean_dice", "mean_med"],
        ascending=[False, False, False, False, True],
    )
    summary.to_csv(output_dir / "profile_summary.csv", index=False)

    best_idx = results.groupby(["case_id", "moving_name"])["match_f1"].idxmax()
    best_per_pair = results.loc[best_idx].sort_values(["case_id", "moving_name"])
    best_per_pair.to_csv(output_dir / "best_per_pair.csv", index=False)

    print("\nTop 15 profiles:")
    print(summary.head(15).to_string(index=False))
    print("\nBest per pair:")
    print(
        best_per_pair[
            [
                "case_id",
                "moving_name",
                "profile",
                "match_f1",
                "matched_mean_iou",
                "mask_dice",
                "centroid_error_median_px",
                "centroid_error_p95_px",
                "inlier_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
