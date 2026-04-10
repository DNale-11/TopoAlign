"""Targeted parameter search for benchmark/run_benchmark.py.

Run a small grid on the worst-performing cases first, then inspect the summary
before validating the best config on a larger subset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = ROOT / "benchmark" / "run_benchmark.py"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark" / "results_tune"


def candidate_configs() -> list[dict]:
    return [
        {
            "name": "best_d_search_ref",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 50,
                "coarse_distance_threshold": 2.0,
                "guided_spatial_window_cap": 60.0,
                "prefer_affine": False,
                "neighbor_weight": 1.0,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 4.0,
                "similarity_residual_threshold": 5.0,
            },
        },
        {
            "name": "best_ref_no_fft",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 50,
                "coarse_distance_threshold": 2.0,
                "guided_spatial_window_cap": 60.0,
                "prefer_affine": False,
                "neighbor_weight": 1.0,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 4.0,
                "similarity_residual_threshold": 5.0,
                "fft_max_iterations": 0,
            },
        },
        {
            "name": "no_fft_neighbor_03",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 50,
                "coarse_distance_threshold": 2.0,
                "guided_spatial_window_cap": 60.0,
                "prefer_affine": False,
                "neighbor_weight": 0.3,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 4.0,
                "similarity_residual_threshold": 5.0,
                "fft_max_iterations": 0,
            },
        },
        {
            "name": "coarse_strict_no_fft",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 30,
                "coarse_distance_threshold": 1.6,
                "guided_spatial_window_cap": 60.0,
                "prefer_affine": False,
                "neighbor_weight": 0.3,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 4.0,
                "similarity_residual_threshold": 5.0,
                "fft_max_iterations": 0,
            },
        },
        {
            "name": "coarse_mid_no_fft_guided90",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 40,
                "coarse_distance_threshold": 1.8,
                "guided_spatial_window_cap": 90.0,
                "prefer_affine": False,
                "neighbor_weight": 0.3,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 4.0,
                "similarity_residual_threshold": 5.0,
                "fft_max_iterations": 0,
            },
        },
        {
            "name": "coarse_mid_no_fft_ransac3",
            "args": {
                "position_weight": 3.5,
                "distance_threshold": 2.2,
                "spatial_window_size": 90.0,
                "top_k": 80,
                "coarse_top_k": 40,
                "coarse_distance_threshold": 1.8,
                "guided_spatial_window_cap": 90.0,
                "prefer_affine": False,
                "neighbor_weight": 0.3,
                "ransac_max_trials": 500,
                "ransac_residual_threshold": 3.0,
                "similarity_residual_threshold": 5.0,
                "fft_max_iterations": 0,
            },
        },
    ]


def build_command(
    output_dir: Path,
    sample_groups: str,
    rounds: str,
    cfg_args: dict,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--sample-groups",
        sample_groups,
        "--rounds",
        rounds,
    ]
    for key, value in cfg_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            cmd.extend([flag, str(value)])
    return cmd


def summarize_results(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    g1d = df[(df["sample_group"] == 1) & (df["round_name"] == "D")]
    summary = {
        "n_pairs": int(len(df)),
        "mean_psnr": float(df["PSNR"].mean()),
        "mean_ssim": float(df["SSIM"].mean()),
        "mean_ncc": float(df["NCC"].mean()),
        "min_ssim": float(df["SSIM"].min()),
        "min_ncc": float(df["NCC"].min()),
        "g1d_ssim": float(g1d["SSIM"].iloc[0]) if len(g1d) == 1 else float("nan"),
        "g1d_ncc": float(g1d["NCC"].iloc[0]) if len(g1d) == 1 else float("nan"),
    }
    summary["score"] = (
        0.8 * summary["mean_ssim"]
        + 0.8 * summary["mean_ncc"]
        + 0.01 * summary["mean_psnr"]
        + 1.2 * summary["min_ncc"]
        + 1.0 * summary["g1d_ncc"]
    )
    return summary


def run_one_config(output_root: Path, sample_groups: str, rounds: str, config: dict) -> dict:
    run_dir = output_root / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(run_dir, sample_groups, rounds, config["args"])
    log_path = run_dir / "tune.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd))
        log.write("\n\n")
        env = dict(os.environ)
        # Force UTF-8 for child process stdout/stderr to avoid cp936/gbk crashes
        # when benchmark prints non-ASCII symbols (e.g. check marks, arrows).
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
            check=False,
        )

    result = {
        "name": config["name"],
        "returncode": int(proc.returncode),
        "output_dir": str(run_dir),
        "log_path": str(log_path),
    }
    if proc.returncode != 0:
        return result

    csv_path = run_dir / "benchmark_results.csv"
    result.update(summarize_results(csv_path))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a targeted benchmark parameter search.")
    parser.add_argument("--sample-groups", default="1,2,3")
    parser.add_argument("--rounds", default="D")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "search")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N candidate configs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = candidate_configs()
    if args.limit is not None:
        configs = configs[: args.limit]

    rows: list[dict] = []
    for idx, config in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] {config['name']}")
        print(json.dumps(config["args"], indent=2))
        result = run_one_config(output_root, args.sample_groups, args.rounds, config)
        rows.append(result)
        if result["returncode"] == 0:
            print(
                "  "
                f"mean_ssim={result['mean_ssim']:.4f} "
                f"mean_ncc={result['mean_ncc']:.4f} "
                f"g1d_ncc={result['g1d_ncc']:.4f} "
                f"min_ncc={result['min_ncc']:.4f} "
                f"score={result['score']:.4f}"
            )
        else:
            print(f"  failed, see {result['log_path']}")

    summary = pd.DataFrame(rows)
    if "score" in summary.columns:
        summary = summary.sort_values(
            by=["score", "g1d_ncc", "min_ncc", "mean_ncc", "mean_ssim"],
            ascending=False,
            kind="stable",
        )
    summary_path = output_root / "tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
