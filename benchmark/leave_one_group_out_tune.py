"""Leave-one-group-out parameter tuning for the registration benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tuning_utils import DEFAULT_OUTPUT_ROOT, run_benchmark, summarize_dataframe, write_json


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
            "name": "joint_loose_seg_similarity",
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
            "name": "joint_dense_seg_similarity",
            "args": {
                "cellpose_diameter": 8.0,
                "cellpose_flow_threshold": -2.0,
                "cellpose_cellprob_threshold": -2.0,
                "cellpose_min_size": 1,
                "position_weight": 4.0,
                "distance_threshold": 2.2,
                "spatial_window_size": 80.0,
                "top_k": 80,
                "validation_min_confidence": 0.15,
                "validation_max_feature_diff": 0.85,
                "prefer_affine": False,
                "ransac_residual_threshold": 4.0,
            },
        },
        {
            "name": "joint_mid_seg_affine",
            "args": {
                "cellpose_diameter": 10.0,
                "cellpose_flow_threshold": -0.5,
                "cellpose_cellprob_threshold": -0.5,
                "cellpose_min_size": 10,
                "position_weight": 3.0,
                "distance_threshold": 2.0,
                "spatial_window_size": 80.0,
                "top_k": 60,
                "validation_min_confidence": 0.25,
                "validation_max_feature_diff": 0.65,
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
    parser = argparse.ArgumentParser(description="Run leave-one-group-out tuning for benchmark parameters.")
    parser.add_argument("--sample-groups", default="1,2,3")
    parser.add_argument("--rounds", default="A,C,D,E")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "leave_one_group_out",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N candidate configs.")
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
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = candidate_configs()
    if args.limit is not None:
        configs = configs[: args.limit]

    overall_rows: list[dict] = []
    holdout_rows: list[dict] = []

    for holdout in sample_groups:
        train_groups = [group for group in sample_groups if group != holdout]
        fold_dir = output_root / f"holdout_g{holdout}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Holdout group {holdout} | train groups {train_groups} ===")

        train_candidates: list[dict] = []
        for idx, config in enumerate(configs, start=1):
            print(f"[train {idx}/{len(configs)}] {config['name']}")
            train_result = run_benchmark(
                output_dir=fold_dir / "train" / config["name"],
                sample_groups=train_groups,
                rounds=rounds,
                cfg_args=config["args"],
                python_executable=args.python_executable,
            )
            row = {
                "holdout_group": holdout,
                "phase": "train",
                "config_name": config["name"],
                "train_groups": ",".join(str(group) for group in train_groups),
                "returncode": train_result["returncode"],
                "output_dir": train_result["output_dir"],
                "log_path": train_result["log_path"],
            }
            if train_result["returncode"] == 0:
                row.update(train_result["summary"])
                train_candidates.append(
                    {
                        "config": config,
                        "summary": train_result["summary"],
                    }
                )
                print(
                    f"  train score={row['score']:.4f} "
                    f"mean_ssim={row['mean_ssim']:.4f} mean_ncc={row['mean_ncc']:.4f}"
                )
            else:
                print(f"  failed, see {row['log_path']}")
            overall_rows.append(row)

        if not train_candidates:
            print(f"No successful train runs for holdout group {holdout}.")
            continue

        train_df = pd.DataFrame([candidate["summary"] | {"config_name": candidate["config"]["name"]} for candidate in train_candidates])
        train_df = train_df.sort_values(
            by=["score", "min_ncc", "mean_ncc", "mean_ssim"],
            ascending=False,
            kind="stable",
        )
        train_df.to_csv(fold_dir / "train_summary.csv", index=False)

        best = max(
            train_candidates,
            key=lambda item: (
                item["summary"]["score"],
                item["summary"]["min_ncc"],
                item["summary"]["mean_ncc"],
                item["summary"]["mean_ssim"],
            ),
        )
        best_config = best["config"]
        print(f"Best train config for holdout G{holdout}: {best_config['name']}")

        holdout_result = run_benchmark(
            output_dir=fold_dir / "holdout" / best_config["name"],
            sample_groups=[holdout],
            rounds=rounds,
            cfg_args=best_config["args"],
            python_executable=args.python_executable,
        )
        holdout_row = {
            "holdout_group": holdout,
            "phase": "holdout",
            "config_name": best_config["name"],
            "train_groups": ",".join(str(group) for group in train_groups),
            "returncode": holdout_result["returncode"],
            "output_dir": holdout_result["output_dir"],
            "log_path": holdout_result["log_path"],
        }
        if holdout_result["returncode"] == 0:
            holdout_row.update(holdout_result["summary"])
            print(
                f"  holdout score={holdout_row['score']:.4f} "
                f"mean_ssim={holdout_row['mean_ssim']:.4f} mean_ncc={holdout_row['mean_ncc']:.4f}"
            )
            holdout_rows.append(holdout_row)
        else:
            print(f"  holdout failed, see {holdout_row['log_path']}")
        overall_rows.append(holdout_row)

        write_json(
            fold_dir / "best_config.json",
            {
                "holdout_group": holdout,
                "train_groups": train_groups,
                "rounds": rounds,
                "best_config_name": best_config["name"],
                "best_config_args": best_config["args"],
                "train_summary": best["summary"],
                "holdout_summary": holdout_result.get("summary"),
            },
        )

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_root / "loo_runs.csv", index=False)
    print(f"\nSaved run log to {output_root / 'loo_runs.csv'}")

    if holdout_rows:
        holdout_df = pd.DataFrame(holdout_rows)
        holdout_df = holdout_df.sort_values("holdout_group", kind="stable")
        holdout_df.to_csv(output_root / "loo_holdout_summary.csv", index=False)
        aggregate = summarize_dataframe(
            holdout_df.rename(
                columns={
                    "mean_psnr": "PSNR",
                    "mean_ssim": "SSIM",
                    "mean_ncc": "NCC",
                    "mean_mi": "MI",
                }
            )[["PSNR", "SSIM", "NCC", "MI"]]
        )
        write_json(output_root / "loo_aggregate_summary.json", aggregate)
        print("\nHoldout summary:")
        print(holdout_df.to_string(index=False))
        print(f"\nAggregate holdout metrics: {aggregate}")


if __name__ == "__main__":
    main()
