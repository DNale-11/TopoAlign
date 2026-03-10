"""Shared helpers for benchmark parameter search scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = ROOT / "benchmark" / "run_benchmark.py"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark" / "results_tune"


def build_command(
    output_dir: Path,
    sample_groups: Iterable[int] | str,
    rounds: Iterable[str] | str,
    cfg_args: dict,
    python_executable: str | None = None,
) -> list[str]:
    if isinstance(sample_groups, str):
        sample_groups_arg = sample_groups
    else:
        sample_groups_arg = ",".join(str(value) for value in sample_groups)

    if isinstance(rounds, str):
        rounds_arg = rounds
    else:
        rounds_arg = ",".join(str(value).upper() for value in rounds)

    cmd = [
        python_executable or sys.executable,
        str(BENCHMARK_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--sample-groups",
        sample_groups_arg,
        "--rounds",
        rounds_arg,
    ]
    for key, value in cfg_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            cmd.extend([flag, str(value)])
    return cmd


def run_benchmark(
    *,
    output_dir: Path,
    sample_groups: Iterable[int] | str,
    rounds: Iterable[str] | str,
    cfg_args: dict,
    python_executable: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        output_dir=output_dir,
        sample_groups=sample_groups,
        rounds=rounds,
        cfg_args=cfg_args,
        python_executable=python_executable,
    )
    log_path = output_dir / "tune.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd))
        log.write("\n\n")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        "returncode": int(proc.returncode),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }
    if proc.returncode != 0:
        return result

    csv_path = output_dir / "benchmark_results.csv"
    result["csv_path"] = str(csv_path)
    result["summary"] = summarize_results(csv_path)
    return result


def summarize_results(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return summarize_dataframe(df)


def summarize_dataframe(df: pd.DataFrame) -> dict:
    summary = {
        "n_pairs": int(len(df)),
        "mean_psnr": float(df["PSNR"].mean()),
        "mean_ssim": float(df["SSIM"].mean()),
        "mean_ncc": float(df["NCC"].mean()),
        "mean_mi": float(df["MI"].mean()),
        "min_ssim": float(df["SSIM"].min()),
        "min_ncc": float(df["NCC"].min()),
    }
    summary["score"] = default_score(summary)
    return summary


def default_score(summary: dict) -> float:
    return float(
        0.9 * summary["mean_ssim"]
        + 0.9 * summary["mean_ncc"]
        + 0.02 * summary["mean_psnr"]
        + 0.8 * summary["min_ssim"]
        + 1.0 * summary["min_ncc"]
    )


def parse_pair_specs(pair_specs: Iterable[str]) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for spec in pair_specs:
        sample_group_raw, round_raw = spec.split(":", maxsplit=1)
        pairs.append((int(sample_group_raw), round_raw.strip().upper()))
    return pairs


def summarize_pairs(df: pd.DataFrame, pairs: Iterable[tuple[int, str]]) -> dict:
    selected = []
    for sample_group, round_name in pairs:
        subset = df[(df["sample_group"] == sample_group) & (df["round_name"] == round_name)]
        if subset.empty:
            continue
        selected.append(subset)
    if not selected:
        return {
            "hard_n_pairs": 0,
            "hard_mean_psnr": float("nan"),
            "hard_mean_ssim": float("nan"),
            "hard_mean_ncc": float("nan"),
            "hard_min_ssim": float("nan"),
            "hard_min_ncc": float("nan"),
            "hard_score": float("-inf"),
        }
    hard_df = pd.concat(selected, ignore_index=True)
    summary = {
        "hard_n_pairs": int(len(hard_df)),
        "hard_mean_psnr": float(hard_df["PSNR"].mean()),
        "hard_mean_ssim": float(hard_df["SSIM"].mean()),
        "hard_mean_ncc": float(hard_df["NCC"].mean()),
        "hard_min_ssim": float(hard_df["SSIM"].min()),
        "hard_min_ncc": float(hard_df["NCC"].min()),
    }
    summary["hard_score"] = float(
        1.2 * summary["hard_mean_ssim"]
        + 1.2 * summary["hard_mean_ncc"]
        + 0.02 * summary["hard_mean_psnr"]
        + 1.0 * summary["hard_min_ssim"]
        + 1.4 * summary["hard_min_ncc"]
    )
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
