"""End-to-end tuning pipeline for robust benchmark parameter selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tuning_utils import (
    DEFAULT_OUTPUT_ROOT,
    run_benchmark,
    summarize_pairs,
    write_json,
)


DEFAULT_HARD_PAIRS = [(1, "A"), (1, "D"), (2, "A"), (3, "A")]


def candidate_configs() -> list[dict]:
    return [
        {
            "name": "baseline_affine",
            "args": {},
        },
        {
            "name": "baseline_similarity",
            "args": {
                "prefer_affine": False,
            },
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
    ]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_round_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def stage1_score(row: dict) -> float:
    return float(0.45 * row["score"] + 0.55 * row["hard_score"])


def stage2_score(summary_df: pd.DataFrame) -> float:
    mean_score = float(summary_df["holdout_score"].mean())
    min_score = float(summary_df["holdout_score"].min())
    mean_ncc = float(summary_df["mean_ncc"].mean())
    min_ncc = float(summary_df["min_ncc"].min())
    return float(0.6 * mean_score + 0.4 * min_score + 0.25 * mean_ncc + 0.35 * min_ncc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full tuning pipeline and final benchmark.")
    parser.add_argument("--sample-groups", default="1,2,3")
    parser.add_argument("--rounds", default="A,C,D,E")
    parser.add_argument("--stage1-rounds", default="A,D")
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "auto_pipeline",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable used to launch benchmark/run_benchmark.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_groups = _parse_int_list(args.sample_groups)
    rounds = _parse_round_list(args.rounds)
    stage1_rounds = _parse_round_list(args.stage1_rounds)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = candidate_configs()
    hard_pairs = [pair for pair in DEFAULT_HARD_PAIRS if pair[0] in sample_groups and pair[1] in stage1_rounds]

    stage1_rows: list[dict] = []
    print("=== Stage 1: Hard-case shortlist ===")
    for idx, config in enumerate(configs, start=1):
        print(f"[stage1 {idx}/{len(configs)}] {config['name']}")
        result = run_benchmark(
            output_dir=output_root / "stage1" / config["name"],
            sample_groups=sample_groups,
            rounds=stage1_rounds,
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
            stage1_rows.append(row)
            continue

        df = pd.read_csv(Path(result["csv_path"]))
        row.update(result["summary"])
        row.update(summarize_pairs(df, hard_pairs))
        row["stage1_score"] = stage1_score(row)
        stage1_rows.append(row)
        print(
            f"  stage1={row['stage1_score']:.4f} "
            f"hard_ssim={row['hard_mean_ssim']:.4f} hard_ncc={row['hard_mean_ncc']:.4f} "
            f"overall_ssim={row['mean_ssim']:.4f}"
        )

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_df = stage1_df.sort_values(
        by=["stage1_score", "hard_min_ncc", "hard_mean_ncc", "mean_ncc", "mean_ssim"],
        ascending=False,
        kind="stable",
    )
    stage1_path = output_root / "stage1_summary.csv"
    stage1_df.to_csv(stage1_path, index=False)
    print(f"Saved stage1 summary to {stage1_path}")

    successful_stage1 = stage1_df[stage1_df["returncode"] == 0].head(args.shortlist_size)
    shortlisted_names = successful_stage1["config_name"].tolist()
    shortlisted = [config for config in configs if config["name"] in shortlisted_names]
    print(f"Shortlisted configs: {shortlisted_names}")

    stage2_rows: list[dict] = []
    print("\n=== Stage 2: Cross-group robustness ===")
    for config in shortlisted:
        fold_rows: list[dict] = []
        print(f"[stage2] {config['name']}")
        for holdout in sample_groups:
            result = run_benchmark(
                output_dir=output_root / "stage2" / config["name"] / f"holdout_g{holdout}",
                sample_groups=[holdout],
                rounds=rounds,
                cfg_args=config["args"],
                python_executable=args.python_executable,
            )
            fold_row = {
                "config_name": config["name"],
                "holdout_group": holdout,
                "returncode": result["returncode"],
                "output_dir": result["output_dir"],
                "log_path": result["log_path"],
            }
            if result["returncode"] != 0:
                print(f"  holdout G{holdout} failed, see {fold_row['log_path']}")
                fold_rows.append(fold_row)
                continue

            fold_row.update(result["summary"])
            fold_row["holdout_score"] = result["summary"]["score"]
            fold_rows.append(fold_row)
            print(
                f"  G{holdout}: score={fold_row['holdout_score']:.4f} "
                f"mean_ssim={fold_row['mean_ssim']:.4f} mean_ncc={fold_row['mean_ncc']:.4f}"
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(output_root / "stage2" / config["name"] / "holdout_summary.csv", index=False)
        ok = fold_df[fold_df["returncode"] == 0].copy()
        if ok.empty:
            continue
        aggregate = {
            "config_name": config["name"],
            "n_holdouts": int(len(ok)),
            "mean_holdout_score": float(ok["holdout_score"].mean()),
            "min_holdout_score": float(ok["holdout_score"].min()),
            "mean_ssim": float(ok["mean_ssim"].mean()),
            "mean_ncc": float(ok["mean_ncc"].mean()),
            "min_ssim": float(ok["min_ssim"].min()),
            "min_ncc": float(ok["min_ncc"].min()),
        }
        aggregate["stage2_score"] = stage2_score(ok)
        stage2_rows.append(aggregate)
        print(
            f"  aggregate stage2={aggregate['stage2_score']:.4f} "
            f"mean_holdout={aggregate['mean_holdout_score']:.4f} "
            f"min_holdout={aggregate['min_holdout_score']:.4f}"
        )

    stage2_df = pd.DataFrame(stage2_rows)
    stage2_df = stage2_df.sort_values(
        by=["stage2_score", "min_ncc", "mean_ncc", "mean_ssim"],
        ascending=False,
        kind="stable",
    )
    stage2_path = output_root / "stage2_summary.csv"
    stage2_df.to_csv(stage2_path, index=False)
    print(f"Saved stage2 summary to {stage2_path}")

    best_name = stage2_df.iloc[0]["config_name"]
    best_config = next(config for config in configs if config["name"] == best_name)
    print(f"\nSelected final config: {best_name}")

    print("\n=== Stage 3: Final full benchmark ===")
    final_result = run_benchmark(
        output_dir=output_root / "final" / best_name,
        sample_groups=sample_groups,
        rounds=rounds,
        cfg_args=best_config["args"],
        python_executable=args.python_executable,
    )
    final_payload = {
        "selected_config_name": best_name,
        "selected_config_args": best_config["args"],
        "stage1_summary_path": str(stage1_path),
        "stage2_summary_path": str(stage2_path),
        "final_output_dir": final_result["output_dir"],
        "final_log_path": final_result["log_path"],
        "final_returncode": final_result["returncode"],
    }
    if final_result["returncode"] == 0:
        final_payload["final_summary"] = final_result["summary"]
        print(f"Final benchmark complete: {final_result['summary']}")
    else:
        print(f"Final benchmark failed, see {final_result['log_path']}")
    write_json(output_root / "final_selection.json", final_payload)


if __name__ == "__main__":
    main()
