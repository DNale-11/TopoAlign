"""Joint Cellpose + matching tuning focused on hard benchmark pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tuning_utils import (
    DEFAULT_OUTPUT_ROOT,
    run_benchmark,
    summarize_pairs,
    summarize_results,
    write_json,
)


DEFAULT_HARD_PAIRS = ["1:A", "1:D", "2:A", "3:A"]


def candidate_configs() -> list[dict]:
    return [
        {
            "name": "baseline_affine",
            "args": {},
        },
        {
            "name": "d_optimized_similarity",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "prefer_affine": False,
                "ransac_residual_threshold": 4.0,
            },
        },
        {
            "name": "loose_seg_low_conf",
            "args": {
                "cellpose_diameter": 8.0,
                "cellpose_flow_threshold": -1.0,
                "cellpose_cellprob_threshold": -1.0,
                "cellpose_min_size": 8,
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "validation_min_confidence": 0.15,
                "validation_max_feature_diff": 0.8,
                "prefer_affine": False,
                "ransac_residual_threshold": 4.0,
            },
        },
        {
            "name": "dense_seg_relaxed_validation",
            "args": {
                "cellpose_diameter": 8.0,
                "cellpose_flow_threshold": -2.0,
                "cellpose_cellprob_threshold": -2.0,
                "cellpose_min_size": 1,
                "position_weight": 4.0,
                "distance_threshold": 2.0,
                "spatial_window_size": 80.0,
                "top_k": 80,
                "validation_min_confidence": 0.1,
                "validation_max_feature_diff": 0.9,
                "prefer_affine": False,
                "ransac_residual_threshold": 4.0,
            },
        },
        {
            "name": "mid_seg_strict_match",
            "args": {
                "cellpose_diameter": 10.0,
                "cellpose_flow_threshold": -0.5,
                "cellpose_cellprob_threshold": -0.5,
                "cellpose_min_size": 12,
                "position_weight": 4.0,
                "distance_threshold": 1.8,
                "spatial_window_size": 70.0,
                "top_k": 60,
                "validation_min_confidence": 0.25,
                "validation_max_feature_diff": 0.65,
                "prefer_affine": False,
                "ransac_residual_threshold": 3.5,
            },
        },
        {
            "name": "small_diameter_affine_guarded",
            "args": {
                "cellpose_diameter": 6.0,
                "cellpose_flow_threshold": -1.0,
                "cellpose_cellprob_threshold": -1.0,
                "cellpose_min_size": 6,
                "position_weight": 3.0,
                "distance_threshold": 2.0,
                "spatial_window_size": 75.0,
                "top_k": 60,
                "validation_min_confidence": 0.2,
                "validation_max_feature_diff": 0.75,
                "prefer_affine": True,
                "ransac_residual_threshold": 3.0,
            },
        },
    ]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_round_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard-case focused joint tuning.")
    parser.add_argument("--sample-groups", default="1,2,3")
    parser.add_argument("--rounds", default="A,D,E")
    parser.add_argument("--hard-pairs", nargs="+", default=DEFAULT_HARD_PAIRS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "hard_case_joint",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N candidate configs.")
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable used to launch benchmark/run_benchmark.py.",
    )
    return parser.parse_args()


def _pair_metrics(df: pd.DataFrame, hard_pairs: list[tuple[int, str]]) -> dict:
    metrics: dict[str, float] = {}
    for sample_group, round_name in hard_pairs:
        key = f"g{sample_group}_{round_name.lower()}"
        subset = df[(df["sample_group"] == sample_group) & (df["round_name"] == round_name)]
        if subset.empty:
            metrics[f"{key}_ssim"] = float("nan")
            metrics[f"{key}_ncc"] = float("nan")
            continue
        metrics[f"{key}_ssim"] = float(subset["SSIM"].iloc[0])
        metrics[f"{key}_ncc"] = float(subset["NCC"].iloc[0])
    return metrics


def main() -> None:
    args = parse_args()
    sample_groups = _parse_int_list(args.sample_groups)
    rounds = _parse_round_list(args.rounds)
    hard_pairs = []
    for item in args.hard_pairs:
        sample_group_raw, round_raw = item.split(":", maxsplit=1)
        hard_pairs.append((int(sample_group_raw), round_raw.strip().upper()))

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = candidate_configs()
    if args.limit is not None:
        configs = configs[: args.limit]

    rows: list[dict] = []
    for idx, config in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] {config['name']}")
        result = run_benchmark(
            output_dir=output_root / config["name"],
            sample_groups=sample_groups,
            rounds=rounds,
            cfg_args=config["args"],
            python_executable=args.python_executable,
        )
        row = {
            "config_name": config["name"],
            "returncode": result["returncode"],
            "output_dir": result["output_dir"],
            "log_path": result["log_path"],
        }
        if result["returncode"] != 0:
            print(f"  failed, see {row['log_path']}")
            rows.append(row)
            continue

        df = pd.read_csv(Path(result["csv_path"]))
        overall = summarize_results(Path(result["csv_path"]))
        hard = summarize_pairs(df, hard_pairs)
        row.update(overall)
        row.update(hard)
        row.update(_pair_metrics(df, hard_pairs))
        row["combined_score"] = float(0.45 * row["score"] + 0.55 * row["hard_score"])
        rows.append(row)
        print(
            f"  combined={row['combined_score']:.4f} "
            f"hard_ssim={row['hard_mean_ssim']:.4f} hard_ncc={row['hard_mean_ncc']:.4f} "
            f"overall_ssim={row['mean_ssim']:.4f} overall_ncc={row['mean_ncc']:.4f}"
        )

    summary = pd.DataFrame(rows)
    if not summary.empty and "combined_score" in summary.columns:
        summary = summary.sort_values(
            by=["combined_score", "hard_min_ncc", "hard_mean_ncc", "mean_ncc", "mean_ssim"],
            ascending=False,
            kind="stable",
        )
    summary_path = output_root / "hard_case_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
        best = summary.iloc[0].to_dict()
        write_json(
            output_root / "best_config_summary.json",
            {
                "sample_groups": sample_groups,
                "rounds": rounds,
                "hard_pairs": [f"{sample_group}:{round_name}" for sample_group, round_name in hard_pairs],
                "best_row": best,
            },
        )


if __name__ == "__main__":
    main()
