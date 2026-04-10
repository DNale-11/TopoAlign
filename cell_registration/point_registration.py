"""Point-set registration utilities for cell centroids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform, estimate_transform

try:
    import napari  # type: ignore
except Exception:
    napari = None


@dataclass
class RegistrationResult:
    transform: AffineTransform
    pts_r1: np.ndarray
    pts_r2: np.ndarray
    pts_r2_reg: np.ndarray


@dataclass
class RobustTransformResult:
    transform: AffineTransform
    inliers: np.ndarray
    residuals: np.ndarray
    method: str

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inliers))

    @property
    def median_inlier_residual(self) -> float:
        if self.inlier_count == 0:
            return float("inf")
        return float(np.median(self.residuals[self.inliers]))

    @property
    def mean_inlier_residual(self) -> float:
        if self.inlier_count == 0:
            return float("inf")
        return float(np.mean(self.residuals[self.inliers]))

    def score(self) -> tuple[int, float, float]:
        return (
            self.inlier_count,
            -self.median_inlier_residual,
            -self.mean_inlier_residual,
        )


def load_data(
    r1_path: str | Path,
    r2_path: str | Path,
    matches_path: str | Path,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load round1, round2, and matches CSVs."""
    r1 = pd.read_csv(r1_path)
    r2 = pd.read_csv(r2_path)
    matches = pd.read_csv(matches_path)
    for col in (x_col, y_col):
        if col not in r1.columns or col not in r2.columns:
            raise ValueError(f"Missing required column '{col}' in input CSVs.")
    # Accept either (cell_id_r1, cell_id_r2) or (cell_id_1, cell_id_2)
    if {"cell_id_r1", "cell_id_r2"}.issubset(matches.columns):
        r1_col, r2_col = "cell_id_r1", "cell_id_r2"
    elif {"cell_id_1", "cell_id_2"}.issubset(matches.columns):
        r1_col, r2_col = "cell_id_1", "cell_id_2"
    else:
        raise ValueError(
            "top_matches.csv must have columns 'cell_id_r1'/'cell_id_r2' (preferred) "
            "or 'cell_id_1'/'cell_id_2' as exported by the main pipeline."
        )
    matches = matches.rename(columns={r1_col: "cell_id_r1", r2_col: "cell_id_r2"})
    return r1, r2, matches


def _select_landmark_matches(matches: pd.DataFrame, top_n: int | None, sort_by: str | None) -> pd.DataFrame:
    """
    Keep only the top-N matches (optionally sorted by a score column) for estimating the transform.
    """
    m = matches
    if sort_by is not None and sort_by in m.columns:
        m = m.sort_values(sort_by, ascending=True)
    if top_n is not None:
        if top_n < 3:
            raise ValueError("Need at least 3 landmark pairs; increase --top-n-landmarks.")
        m = m.head(top_n)
    return m.reset_index(drop=True)


def _gather_landmarks(
    r1: pd.DataFrame, r2: pd.DataFrame, matches: pd.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build landmark arrays aligned by matches."""
    pts_r1 = []
    pts_r2 = []
    for _, row in matches.iterrows():
        cid1 = row["cell_id_r1"]
        cid2 = row["cell_id_r2"]
        rec1 = r1.loc[r1["cell_id"] == cid1]
        rec2 = r2.loc[r2["cell_id"] == cid2]
        if rec1.empty or rec2.empty:
            continue
        pts_r1.append([rec1.iloc[0][x_col], rec1.iloc[0][y_col]])
        pts_r2.append([rec2.iloc[0][x_col], rec2.iloc[0][y_col]])
    if len(pts_r1) < 3:
        raise ValueError("Need at least 3 matched landmarks to estimate a transform.")
    return np.asarray(pts_r1, dtype=float), np.asarray(pts_r2, dtype=float)


def _as_affine_transform(transform) -> AffineTransform:
    return AffineTransform(matrix=np.asarray(transform.params, dtype=float))


def _point_residuals(transform: AffineTransform, pts_moving: np.ndarray, pts_fixed: np.ndarray) -> np.ndarray:
    pts_reg = transform(pts_moving)
    return np.linalg.norm(pts_reg - pts_fixed, axis=1)


def estimate_robust_transform(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    *,
    prefer_affine: bool = False,
    allow_scale: bool = False,
    residual_threshold: float = 3.0,
    similarity_residual_threshold: float | None = None,
    max_trials: int = 500,
    min_inliers: int = 3,
    fallback_to_translation: bool = True,
) -> RobustTransformResult:
    if len(pts_fixed) != len(pts_moving):
        raise ValueError("Point arrays must have the same length.")
    if len(pts_fixed) < 3:
        raise ValueError("Need at least 3 matched landmarks to estimate a transform.")

    similarity_threshold = (
        residual_threshold if similarity_residual_threshold is None else similarity_residual_threshold
    )

    candidates: list[tuple[str, str, type[AffineTransform], int, float]] = []
    if not allow_scale:
        candidates.append(("rigid", "euclidean", EuclideanTransform, 2, residual_threshold))
    else:
        if prefer_affine and len(pts_fixed) >= 4:
            candidates.append(("affine", "affine", AffineTransform, 4, residual_threshold))
        candidates.append(("similarity", "similarity", SimilarityTransform, 3, similarity_threshold))

    best: RobustTransformResult | None = None
    for display_name, estimate_method, model_cls, min_samples, threshold in candidates:
        try:
            model_robust, inliers = ransac(
                (pts_moving, pts_fixed),
                model_cls,
                min_samples=min_samples,
                residual_threshold=threshold,
                max_trials=max_trials,
            )
        except Exception:
            continue

        if model_robust is None or inliers is None or int(inliers.sum()) < max(min_inliers, min_samples):
            continue

        try:
            refit = estimate_transform(estimate_method, pts_moving[inliers], pts_fixed[inliers])
            transform = _as_affine_transform(refit)
        except Exception:
            transform = _as_affine_transform(model_robust)

        residuals = _point_residuals(transform, pts_moving, pts_fixed)
        candidate = RobustTransformResult(
            transform=transform,
            inliers=np.asarray(inliers, dtype=bool),
            residuals=residuals,
            method=display_name,
        )
        if best is None or candidate.score() > best.score():
            best = candidate

    if best is not None:
        return best

    if not fallback_to_translation:
        raise RuntimeError("RANSAC failed to find a valid transform.")

    translation = np.median(pts_fixed - pts_moving, axis=0)
    transform = AffineTransform(translation=(float(translation[0]), float(translation[1])))
    residuals = _point_residuals(transform, pts_moving, pts_fixed)
    inliers = residuals <= max(residual_threshold, similarity_threshold)
    return RobustTransformResult(
        transform=transform,
        inliers=inliers,
        residuals=residuals,
        method="translation",
    )


def estimate_initial_transform(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    method: Literal[ "similarity", "affine"] = "similarity",
    use_ransac: bool = True,
) -> AffineTransform:
    """Estimate a global transform mapping r2 -> r1 from landmarks."""
    if method == "similarity":
        model_cls = SimilarityTransform
    elif method == "affine":
        model_cls = AffineTransform
    else:
        raise ValueError("method must be'similarity', or 'affine'")

    if use_ransac:
        model_robust, inliers = ransac(
            (pts_r2, pts_r1),
            model_cls,
            min_samples=3,
            residual_threshold=5.0,
            max_trials=100,
        )
        if inliers is None or inliers.sum() < 3:
            raise RuntimeError("RANSAC failed to find a valid transform.")
        return model_robust  # type: ignore

    return estimate_transform(method, pts_r2, pts_r1)  # type: ignore


def apply_transform(df: pd.DataFrame, transform: AffineTransform, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """Apply transform to all centroids in df, adding x_reg/y_reg columns."""
    pts = df[[x_col, y_col]].to_numpy(dtype=float)
    pts_reg = transform(pts)
    df_out = df.copy()
    df_out["x_reg"] = pts_reg[:, 0]
    df_out["y_reg"] = pts_reg[:, 1]
    return df_out


def _similarity_from_params(params: np.ndarray) -> AffineTransform:
    """Build a similarity transform from params=[theta, tx, ty, log_scale]."""
    theta, tx, ty, log_s = params
    s = np.exp(log_s)
    c, sgn = np.cos(theta), np.sin(theta)
    matrix = np.array(
        [
            [s * c, -s * sgn, tx],
            [s * sgn, s * c, ty],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return AffineTransform(matrix=matrix)


def _neighbor_distance_vectors(pts: np.ndarray, k: int) -> list[np.ndarray]:
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=min(k + 1, len(pts)))  # include self at idx 0
    # drop self distance at 0
    return [d[1:] for d in dists]


def refine_transform_with_neighbors(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    initial: AffineTransform,
    k: int = 5,
    max_iter: int = 80,
    neighbor_weight: float = 1.0,
    landmark_weight: float = 10.0,
    max_theta_deg: float = 10.0,
    max_translation: float | None = 30.0,
    max_scale_change: float = 0.05,
    accept_worse: float = 1.05,
) -> AffineTransform:
    """
    Small refinement: adjust similarity params to preserve neighbor distances
    around matched landmarks. Returns refined transform.

    Landmark alignment is given a strong weight so top matches stay aligned, and
    parameter updates are clamped to small rotations/scales to avoid flips.
    """
    if len(pts_r1) < 3:
        return initial

    def _landmark_losses(pts_r2_reg: np.ndarray) -> tuple[float, float]:
        diff = pts_r1 - pts_r2_reg
        per_point = np.sum(diff * diff, axis=1)
        return float(np.mean(per_point)), float(np.median(per_point))

    def _neighbor_loss(pts_r2_reg: np.ndarray) -> float:
        nb1 = _neighbor_distance_vectors(pts_r1, k=k)
        nb2 = _neighbor_distance_vectors(pts_r2_reg, k=k)
        diffs = []
        for d1, d2 in zip(nb1, nb2):
            m = min(len(d1), len(d2))
            if m == 0:
                continue
            diffs.append(((d1[:m] - d2[:m]) ** 2).mean())
        if not diffs:
            return np.inf
        return float(np.mean(diffs))

    def _loss_components(params: np.ndarray) -> tuple[float, float]:
        T = _similarity_from_params(params)
        pts_r2_reg = T(pts_r2)
        lm_mean, _ = _landmark_losses(pts_r2_reg)
        return lm_mean, _neighbor_loss(pts_r2_reg)

    base_mean, base_median = _landmark_losses(initial(pts_r2))

    # seed params from initial similarity approximation
    A = initial.params
    s_est = np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)
    theta_est = np.arctan2(A[1, 0] / max(s_est, 1e-8), A[0, 0] / max(s_est, 1e-8))
    tx_est, ty_est = A[0, 2], A[1, 2]
    x0 = np.array([theta_est, tx_est, ty_est, np.log(s_est + 1e-8)], dtype=float)

    # Clamp updates around the initial estimate.
    theta_margin = np.deg2rad(max_theta_deg)
    if max_translation is None:
        tx_bounds = (-np.inf, np.inf)
        ty_bounds = (-np.inf, np.inf)
    else:
        tx_bounds = (tx_est - max_translation, tx_est + max_translation)
        ty_bounds = (ty_est - max_translation, ty_est + max_translation)
    s_lower = max(1e-4, s_est * (1.0 - max_scale_change))
    s_upper = s_est * (1.0 + max_scale_change)
    bounds = (
        (theta_est - theta_margin, theta_est + theta_margin),
        tx_bounds,
        ty_bounds,
        (np.log(s_lower), np.log(s_upper)),
    )

    def objective(params: np.ndarray) -> float:
        landmark_loss, nb_loss = _loss_components(params)
        if np.isinf(nb_loss):
            nb_loss = 1e6
        return float(landmark_weight * landmark_loss + neighbor_weight * nb_loss)

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "disp": False},
    )
    if not res.success:
        return initial

    candidate = _similarity_from_params(res.x)
    cand_mean, _ = _loss_components(res.x)
    cand_mean_full, cand_median_full = _landmark_losses(candidate(pts_r2))
    if cand_mean_full > base_mean * accept_worse or cand_median_full > base_median * accept_worse:
        return initial
    return candidate


def plot_registration(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    pts_r2_reg: np.ndarray,
    out_path: str | Path | None = None,
) -> None:
    """Simple scatter/connection plot of landmark alignment."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts_r1[:, 0], pts_r1[:, 1], c="dodgerblue", label="Round1 landmarks", s=20)
    ax.scatter(pts_r2_reg[:, 0], pts_r2_reg[:, 1], c="tomato", label="Round2 registered", s=20)
    for a, b in zip(pts_r1, pts_r2_reg):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", alpha=0.6, linewidth=0.8)
    ax.legend()
    ax.set_aspect("equal")
    ax.invert_yaxis()  # common for image coordinates
    ax.set_title("Landmark alignment (r2 -> r1)")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def run_registration(
    r1_csv: str | Path,
    r2_csv: str | Path,
    matches_csv: str | Path,
    output_registered_csv: str | Path = "round2_cells_registered.csv",
    plot_path: str | Path | None = "registration_plot.png",
    napari_view: bool = False,
    method: Literal["similarity", "affine"] = "similarity",
    use_ransac: bool = True,
    refine: bool = True,
    neighbor_k: int = 5,
    neighbor_weight: float = 1.0,
    landmark_weight: float = 10.0,
    top_n_landmarks: int | None = None,
    sort_by: str | None = "distance",
    max_rotation_deg: float = 10.0,
    max_scale_change: float = 0.05,
    max_translation: float | None = 30.0,
    accept_worse: float = 1.05,
) -> RegistrationResult:
    """Full pipeline: load, estimate transform, optional refine, apply, and save."""
    r1, r2, matches = load_data(r1_csv, r2_csv, matches_csv, x_col="x", y_col="y")
    matches_landmarks = _select_landmark_matches(matches, top_n=top_n_landmarks, sort_by=sort_by)
    pts_r1, pts_r2 = _gather_landmarks(r1, r2, matches_landmarks, x_col="x", y_col="y")

    T_init = estimate_initial_transform(pts_r1, pts_r2, method=method, use_ransac=use_ransac)
    T_final = T_init
    if refine and method == "similarity":
        T_final = refine_transform_with_neighbors(
            pts_r1,
            pts_r2,
            initial=T_init,
            k=neighbor_k,
            neighbor_weight=neighbor_weight,
            landmark_weight=landmark_weight,
            max_theta_deg=max_rotation_deg,
            max_translation=max_translation,
            max_scale_change=max_scale_change,
            accept_worse=accept_worse,
        )

    pts_r2_reg = T_final(pts_r2)
    r2_reg = apply_transform(r2, T_final, x_col="x", y_col="y")
    Path(output_registered_csv).parent.mkdir(parents=True, exist_ok=True)
    r2_reg.to_csv(output_registered_csv, index=False)

    if plot_path is not None:
        plot_registration(pts_r1, pts_r2, pts_r2_reg, out_path=plot_path)

    if napari_view:
        if napari is None:
            print("napari not installed; skipping napari view.")
        else:
            all_r1 = r1[["x", "y"]].to_numpy(float)
            all_r2 = r2[["x", "y"]].to_numpy(float)
            all_r2_reg = T_final(all_r2)

            viewer = napari.Viewer(title="Registration (all cells)")
            # All cells
            viewer.add_points(all_r1, name="round1 all", face_color="dodgerblue", size=4, opacity=0.5)
            viewer.add_points(all_r2, name="round2 all (pre)", face_color="orange", size=4, opacity=0.3)
            viewer.add_points(all_r2_reg, name="round2 all (reg)", face_color="red", size=4, opacity=0.6)
            # Landmarks for reference
            viewer.add_points(pts_r1, name="landmarks r1", face_color="cyan", size=8)
            viewer.add_points(pts_r2, name="landmarks r2 (pre)", face_color="yellow", size=8, opacity=0.6)
            viewer.add_points(pts_r2_reg, name="landmarks r2 (reg)", face_color="magenta", size=9, opacity=0.9)
            viewer.window._qt_window.raise_()
            napari.run()

    return RegistrationResult(transform=T_final, pts_r1=pts_r1, pts_r2=pts_r2, pts_r2_reg=pts_r2_reg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register round2 nuclei coordinates into round1 space.")
    parser.add_argument("round1_csv", type=Path, help="CSV with round1 cells (columns: cell_id,x,y, ...)")
    parser.add_argument("round2_csv", type=Path, help="CSV with round2 cells (columns: cell_id,x,y, ...)")
    parser.add_argument("top_matches_csv", type=Path, help="CSV with columns cell_id_r1, cell_id_r2, (score optional).")
    parser.add_argument("--output-registered-csv", type=Path, default="round2_cells_registered.csv")
    parser.add_argument("--plot-path", type=Path, default="registration_plot.png", help="If set, save landmark alignment plot.")
    parser.add_argument(
        "--method",
        choices=["similarity", "affine"],
        default="similarity",
        help="Transform model to fit from landmarks.",
    )
    parser.add_argument("--no-ransac", action="store_true", help="Disable RANSAC; fit directly to all landmarks.")
    parser.add_argument("--no-refine", action="store_true", help="Disable neighbor-based refinement.")
    parser.add_argument("--neighbor-k", type=int, default=5, help="k-NN used in refinement diagnostics.")
    parser.add_argument("--neighbor-weight", type=float, default=1.0, help="Weight of neighbor distance loss.")
    parser.add_argument(
        "--landmark-weight",
        type=float,
        default=10.0,
        help="Weight to keep top landmark pairs aligned during refinement.",
    )
    parser.add_argument(
        "--top-n-landmarks",
        type=int,
        default=None,
        help="Use only the best N matches (sorted by --sort-by) to estimate the transform.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="distance",
        help="Column in matches CSV to sort by before selecting top landmarks (default: distance ascending).",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=10.0,
        help="Maximum rotation change (degrees) allowed during refinement.",
    )
    parser.add_argument(
        "--max-scale-change",
        type=float,
        default=0.05,
        help="Maximum relative scale change allowed during refinement (0.05 = ±5%).",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=30.0,
        help="Maximum translation change (pixels) allowed during refinement.",
    )
    parser.add_argument(
        "--accept-worse",
        type=float,
        default=1.05,
        help="Reject refinement if landmark MSE worsens beyond this factor vs. the initial transform.",
    )
    parser.add_argument("--napari", action="store_true", help="Open napari viewer to inspect landmark alignment.")
    args = parser.parse_args()

    run_registration(
        r1_csv=args.round1_csv,
        r2_csv=args.round2_csv,
        matches_csv=args.top_matches_csv,
        output_registered_csv=args.output_registered_csv,
        plot_path=args.plot_path,
        method=args.method,
        use_ransac=not args.no_ransac,
        refine=not args.no_refine,
        neighbor_k=args.neighbor_k,
        neighbor_weight=args.neighbor_weight,
        landmark_weight=args.landmark_weight,
        top_n_landmarks=args.top_n_landmarks,
        sort_by=args.sort_by,
        max_rotation_deg=args.max_rotation_deg,
        max_scale_change=args.max_scale_change,
        max_translation=args.max_translation,
        accept_worse=args.accept_worse,
        napari_view=args.napari,
    )
