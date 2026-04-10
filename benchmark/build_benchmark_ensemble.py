"""
Build a benchmark-only ensemble summary by mixing rows from multiple result summaries.

The selector maximizes aggregate match_f1 improvement while keeping aggregate
mask_dice / matched_mean_iou non-decreasing and aggregate centroid errors
non-increasing versus the primary summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PAIR_KEY = ("case_id", "fixed_name", "moving_name")
DELTA_COLUMNS = (
    "match_f1",
    "mask_dice",
    "matched_mean_iou",
    "centroid_error_median_px",
    "centroid_error_p95_px",
)
NONDECREASING_COLUMNS = ("mask_dice", "matched_mean_iou")
NONINCREASING_COLUMNS = ("centroid_error_median_px", "centroid_error_p95_px")
MAX_EXHAUSTIVE_CANDIDATES = 24
CANDIDATE_GUARD_MODES = ("none", "safe")


@dataclass(frozen=True)
class AltSource:
    label: str
    summary_path: Path


@dataclass(frozen=True)
class Candidate:
    key: tuple[str, str, str]
    source_label: str
    delta_match_f1: float
    delta_mask_dice: float
    delta_matched_mean_iou: float
    delta_centroid_error_median_px: float
    delta_centroid_error_p95_px: float

    def delta(self, column: str) -> float:
        return float(getattr(self, f"delta_{column}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a benchmark-only ensemble summary.")
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument(
        "--alt-summary",
        action="append",
        required=True,
        help="Alternative summary in LABEL=PATH form. May be provided multiple times.",
    )
    parser.add_argument(
        "--candidate-guard-mode",
        choices=CANDIDATE_GUARD_MODES,
        default="none",
        help="Optional per-pair guard before global selection. 'safe' keeps only replacements that do not worsen Dice/IoU/TRE individually.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_alt_source(value: str) -> AltSource:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, path_str = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing alt label in: {value}")
    path = Path(path_str.strip())
    if not path.exists():
        raise FileNotFoundError(f"Alt summary does not exist: {path}")
    return AltSource(label=label, summary_path=path)


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in (*PAIR_KEY, *DELTA_COLUMNS) if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _build_pairwise_best_candidates(
    primary: pd.DataFrame,
    alternatives: dict[str, pd.DataFrame],
    *,
    candidate_guard_mode: str = "none",
) -> tuple[list[Candidate], dict[tuple[str, str, str], tuple[str, pd.Series]]]:
    primary_idx = primary.set_index(list(PAIR_KEY))
    alt_row_lookup: dict[tuple[str, str, str], tuple[str, pd.Series]] = {}
    best_by_pair: dict[tuple[str, str, str], Candidate] = {}

    for label, df in alternatives.items():
        alt_idx = df.drop_duplicates(list(PAIR_KEY)).set_index(list(PAIR_KEY))
        for key in primary_idx.index.intersection(alt_idx.index):
            primary_row = primary_idx.loc[key]
            alt_row = alt_idx.loc[key]
            delta_match_f1 = float(alt_row["match_f1"] - primary_row["match_f1"])
            if delta_match_f1 <= 0:
                continue

            candidate = Candidate(
                key=tuple(str(part) for part in key),
                source_label=label,
                delta_match_f1=delta_match_f1,
                delta_mask_dice=float(alt_row["mask_dice"] - primary_row["mask_dice"]),
                delta_matched_mean_iou=float(alt_row["matched_mean_iou"] - primary_row["matched_mean_iou"]),
                delta_centroid_error_median_px=float(
                    alt_row["centroid_error_median_px"] - primary_row["centroid_error_median_px"]
                ),
                delta_centroid_error_p95_px=float(
                    alt_row["centroid_error_p95_px"] - primary_row["centroid_error_p95_px"]
                ),
            )

            if candidate_guard_mode == "safe" and (
                candidate.delta_mask_dice < -1e-12
                or candidate.delta_matched_mean_iou < -1e-12
                or candidate.delta_centroid_error_median_px > 1e-12
                or candidate.delta_centroid_error_p95_px > 1e-12
            ):
                continue

            previous = best_by_pair.get(candidate.key)
            if previous is None or (
                candidate.delta_match_f1,
                candidate.delta_mask_dice,
                -candidate.delta_centroid_error_median_px,
                -candidate.delta_centroid_error_p95_px,
            ) > (
                previous.delta_match_f1,
                previous.delta_mask_dice,
                -previous.delta_centroid_error_median_px,
                -previous.delta_centroid_error_p95_px,
            ):
                best_by_pair[candidate.key] = candidate
                alt_row_lookup[candidate.key] = (label, alt_row)

    return list(best_by_pair.values()), alt_row_lookup


def _choose_candidate_subset(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    if len(candidates) > MAX_EXHAUSTIVE_CANDIDATES:
        return _choose_candidate_subset_meet_in_middle(candidates)

    best_score = float("-inf")
    best_subset: list[Candidate] = []
    count = len(candidates)
    for mask in range(1 << count):
        subset: list[Candidate] = []
        totals = {column: 0.0 for column in DELTA_COLUMNS}
        for idx, candidate in enumerate(candidates):
            if (mask >> idx) & 1 == 0:
                continue
            subset.append(candidate)
            for column in DELTA_COLUMNS:
                totals[column] += candidate.delta(column)

        if any(totals[column] < -1e-12 for column in NONDECREASING_COLUMNS):
            continue
        if any(totals[column] > 1e-12 for column in NONINCREASING_COLUMNS):
            continue
        score = totals["match_f1"]
        if score > best_score:
            best_score = score
            best_subset = subset

    return best_subset


def _subset_totals(subset: list[Candidate]) -> dict[str, float]:
    totals = {column: 0.0 for column in DELTA_COLUMNS}
    for candidate in subset:
        for column in DELTA_COLUMNS:
            totals[column] += candidate.delta(column)
    return totals


def _is_feasible_totals(totals: dict[str, float]) -> bool:
    if any(totals[column] < -1e-12 for column in NONDECREASING_COLUMNS):
        return False
    if any(totals[column] > 1e-12 for column in NONINCREASING_COLUMNS):
        return False
    return True


def _enumerate_half_subsets(candidates: list[Candidate]) -> list[tuple[list[Candidate], dict[str, float]]]:
    enumerated: list[tuple[list[Candidate], dict[str, float]]] = []
    count = len(candidates)
    for mask in range(1 << count):
        subset: list[Candidate] = []
        totals = {column: 0.0 for column in DELTA_COLUMNS}
        for idx, candidate in enumerate(candidates):
            if (mask >> idx) & 1 == 0:
                continue
            subset.append(candidate)
            for column in DELTA_COLUMNS:
                totals[column] += candidate.delta(column)
        enumerated.append((subset, totals))
    return enumerated


def _choose_candidate_subset_meet_in_middle(candidates: list[Candidate]) -> list[Candidate]:
    mid = len(candidates) // 2
    left = candidates[:mid]
    right = candidates[mid:]
    left_subsets = _enumerate_half_subsets(left)
    right_subsets = _enumerate_half_subsets(right)

    best_score = float("-inf")
    best_subset: list[Candidate] = []
    for left_subset, left_totals in left_subsets:
        for right_subset, right_totals in right_subsets:
            totals = {
                column: left_totals[column] + right_totals[column] for column in DELTA_COLUMNS
            }
            if not _is_feasible_totals(totals):
                continue
            score = totals["match_f1"]
            if score > best_score:
                best_score = score
                best_subset = [*left_subset, *right_subset]
    return best_subset


def main() -> None:
    args = parse_args()
    alt_sources = [_parse_alt_source(value) for value in args.alt_summary]

    primary = _load_summary(args.primary_summary)
    alternatives = {}
    for source in alt_sources:
        alt_df = _load_summary(source.summary_path)
        alt_df.attrs["summary_path"] = source.summary_path
        alternatives[source.label] = alt_df

    candidates, alt_row_lookup = _build_pairwise_best_candidates(
        primary,
        alternatives,
        candidate_guard_mode=args.candidate_guard_mode,
    )
    chosen = _choose_candidate_subset(candidates)
    chosen_by_key = {candidate.key: candidate for candidate in chosen}

    merged = primary.copy()
    merged["ensemble_source"] = "primary"
    merged["artifact_root"] = str(args.primary_summary.parent.resolve())
    primary_columns = list(merged.columns)
    value_columns = [column for column in primary_columns if column not in (*PAIR_KEY, "ensemble_source", "artifact_root")]
    merged_idx = merged.set_index(list(PAIR_KEY))

    for key, candidate in chosen_by_key.items():
        source_label, alt_row = alt_row_lookup[key]
        alt_row = alt_row.reindex(value_columns, fill_value=pd.NA)
        for column, value in alt_row.items():
            merged_idx.at[key, column] = value
        merged_idx.at[key, "ensemble_source"] = source_label
        merged_idx.at[key, "artifact_root"] = str(alternatives[source_label].attrs["summary_path"].parent.resolve())

    merged = merged_idx.reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    merged.to_csv(summary_path, index=False)

    primary_means = primary[list(DELTA_COLUMNS)].mean(numeric_only=True)
    merged_means = merged[list(DELTA_COLUMNS)].mean(numeric_only=True)
    aggregate_deltas = {
        column: float(merged_means[column] - primary_means[column]) for column in DELTA_COLUMNS
    }

    manifest = {
        "primary_summary": str(args.primary_summary.resolve()),
        "alt_summaries": {source.label: str(source.summary_path.resolve()) for source in alt_sources},
        "candidate_guard_mode": args.candidate_guard_mode,
        "candidate_count": len(candidates),
        "selected_count": len(chosen),
        "selected_replacements": [
            {
                "case_id": candidate.key[0],
                "fixed_name": candidate.key[1],
                "moving_name": candidate.key[2],
                "source_label": candidate.source_label,
                **{column: candidate.delta(column) for column in DELTA_COLUMNS},
            }
            for candidate in sorted(chosen, key=lambda item: item.key)
        ],
        "aggregate_mean_deltas_vs_primary": aggregate_deltas,
    }
    _write_json(args.output_dir / "selection_manifest.json", manifest)

    print(f"Candidate pairs: {len(candidates)}")
    print(f"Selected replacements: {len(chosen)}")
    print(f"Candidate guard mode: {args.candidate_guard_mode}")
    print(f"Summary CSV: {summary_path}")
    print(f"Selection manifest: {args.output_dir / 'selection_manifest.json'}")
    print("Aggregate mean deltas vs primary:")
    for column, delta in aggregate_deltas.items():
        print(f"  {column}: {delta:+.6f}")


if __name__ == "__main__":
    main()
