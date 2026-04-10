from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from .config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .matching import MatchingConfig, greedy_match_cells, match_cells_per_patch, match_cells_per_cluster
from .registration import (
    RigidTransform,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
)
from .robust_alignment import perform_global_registration, apply_transform_to_coordinates
from .segmentation import CellposeSegmenter
from .io_utils import infer_image_mode, load_image, project_intensity_max

# Allow running as `python cell_registration/main.py` by setting package context.
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "cell_registration"

# Single source of truth for defaults.
DEFAULT_IMG1_PATH = Path("1_ch5.tif")
DEFAULT_IMG2_PATH = Path("2_ch5.tif")
DEFAULT_TOP_K = 50
DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_SAVE_MATCH_TABLE = Path("outputs/top_matches.csv")
DEFAULT_SAVE_MATCH_OVERLAY = Path("outputs/match_overlay")
DEFAULT_SAVE_SEGMENTATION_PREFIX = Path("outputs/segmentation")
DEFAULT_SAVE_MATCH_PLOT = Path("outputs/match_plot")
DEFAULT_SAVE_REGISTRATION_OVERLAY = Path("outputs/registration_overlay")
DEFAULT_SAVE_FEATURES_DIR = Path("outputs")
DEFAULT_RESIDUAL_PRUNE_QUANTILE = 0.9
MIN_MATCHES_FOR_REFINEMENT = 3
TOP_K_PER_PATCH = 6  # default 6 per patch -> 54 for 3x3 grid

PATCH_GRID = 3  # 3x3 patches across the image
PATCH_W = 1.0 / PATCH_GRID
PATCH_H = 1.0 / PATCH_GRID

def assign_patches(
    df: pd.DataFrame,
    image_width: float,
    image_height: float,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """Attach normalized coordinates and 3x3 patch indices to the cell table."""
    out = df.copy()
    out["x_norm"] = (out[x_col] / float(image_width)).clip(0.0, 1.0)
    out["y_norm"] = (out[y_col] / float(image_height)).clip(0.0, 1.0)
    out["patch_x"] = np.clip((out["x_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
    out["patch_y"] = np.clip((out["y_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
    out["patch_id"] = list(zip(out["patch_y"], out["patch_x"]))
    return out


def assign_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Assign each cell to a spatial cluster using KMeans on (x_col, y_col).

    Operates on a copy of the dataframe and returns it with the new cluster column.
    """
    out = df.copy()
    n_eff = min(n_clusters, len(out))
    if n_eff <= 1:
        out[cluster_col] = 0
        return out

    coords = out[[x_col, y_col]].to_numpy()
    kmeans = KMeans(n_clusters=n_eff, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(coords)
    out[cluster_col] = labels
    return out


def assign_clusters_from_round1(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    n_clusters: int,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    cluster_col: str = "cluster_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit KMeans on round1 cell coordinates and assign cluster labels to both rounds.
    """
    feats1_out = feats1.copy()
    feats2_out = feats2.copy()
    n_eff = min(n_clusters, len(feats1_out))
    if n_eff <= 1:
        feats1_out[cluster_col] = 0
        feats2_out[cluster_col] = 0
        return feats1_out, feats2_out

    coords1 = feats1_out[[x_col, y_col]].to_numpy()
    kmeans = KMeans(n_clusters=n_eff, random_state=0, n_init="auto")
    kmeans.fit(coords1)
    feats1_out[cluster_col] = kmeans.labels_
    if len(feats2_out) == 0:
        feats2_out[cluster_col] = pd.Series(dtype=int)
    else:
        coords2 = feats2_out[[x_col, y_col]].to_numpy()
        feats2_out[cluster_col] = kmeans.predict(coords2)
    return feats1_out, feats2_out

def load_cells(
    csv_path: Path | str,
    image_width: float,
    image_height: float,
    id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """Load a cell CSV and append normalized coordinates plus patch labels."""
    df = pd.read_csv(csv_path)
    for required in (id_col, x_col, y_col):
        if required not in df.columns:
            raise ValueError(f"Missing required column '{required}' in {csv_path}")
    return assign_patches(df, image_width, image_height, x_col=x_col, y_col=y_col)


def _patch_center(patch_x: int, patch_y: int) -> tuple[float, float]:
    """Return normalized patch center coordinates for a given patch index."""
    x0 = patch_x / PATCH_GRID
    y0 = patch_y / PATCH_GRID
    return x0 + PATCH_W / 2.0, y0 + PATCH_H / 2.0


def _patch_diag_px(image_width: float, image_height: float) -> float:
    """Compute the diagonal length (in pixels) of a single patch."""
    patch_w_px = image_width / PATCH_GRID
    patch_h_px = image_height / PATCH_GRID
    return math.hypot(patch_w_px, patch_h_px)


def _apply_rigid_to_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform (2x2 R, 2-dim t) to an array of (x,y) points."""
    return (rotation @ points.T).T + translation


def compute_L_pos(
    cell_r1: pd.Series,
    cell_r2: pd.Series,
    image_width: float,
    image_height: float,
) -> float:
    """Position loss between two cells inside the same patch."""
    _ = (image_width, image_height)  # kept for signature clarity, not needed because normalized coordinates are patch-agnostic
    cx, cy = _patch_center(int(cell_r1["patch_x"]), int(cell_r1["patch_y"]))
    rx1 = (cell_r1["x_norm"] - cx) / (PATCH_W / 2.0)
    ry1 = (cell_r1["y_norm"] - cy) / (PATCH_H / 2.0)
    rx2 = (cell_r2["x_norm"] - cx) / (PATCH_W / 2.0)
    ry2 = (cell_r2["y_norm"] - cy) / (PATCH_H / 2.0)
    dx = rx1 - rx2
    dy = ry1 - ry2
    return math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)


def _knn_distances(
    df_patch: pd.DataFrame,
    anchor: pd.Series,
    k: int,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    """Return the k-NN distances (pixels) around anchor within the given patch."""
    if len(df_patch) <= 1:
        return np.array([])
    coords = df_patch[[x_col, y_col]].to_numpy()
    query = np.array([[anchor[x_col], anchor[y_col]]], dtype=float)
    k_eff = min(k + 1, len(df_patch))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree")
    nn.fit(coords)
    dists, _ = nn.kneighbors(query, return_distance=True)
    return dists[0][1:]  # drop self


def compute_L_nei(
    cell_r1: pd.Series,
    cell_r2: pd.Series,
    df_r1_patch: pd.DataFrame,
    df_r2_patch: pd.DataFrame,
    k: int,
    image_width: float,
    image_height: float,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> float:
    """Neighbor structure loss using k-NN distance vectors in the same patch."""
    k_eff = min(k, len(df_r1_patch) - 1, len(df_r2_patch) - 1)
    if k_eff <= 0:
        return float("inf")
    d1 = _knn_distances(df_r1_patch, cell_r1, k_eff, x_col=x_col, y_col=y_col)
    d2 = _knn_distances(df_r2_patch, cell_r2, k_eff, x_col=x_col, y_col=y_col)
    if len(d1) != len(d2) or len(d1) == 0:
        return float("inf")
    diag = _patch_diag_px(image_width, image_height)
    d1_norm = d1 / diag
    d2_norm = d2 / diag
    diff = d1_norm - d2_norm
    mse = float(np.mean(diff**2))
    return float(np.sqrt(mse))


def filter_top_pairs(
    candidate_matches: pd.DataFrame,
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    k: int,
    image_width: float,
    image_height: float,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """Filter candidate matches by position and neighborhood similarity."""
    if candidate_matches.empty:
        return pd.DataFrame(
            columns=[id_r1_col, id_r2_col, "L_pos", "L_nei", "patch_x", "patch_y"]
        )
    r1_lookup = df_r1.set_index(cell_id_col)
    r2_lookup = df_r2.set_index(cell_id_col)
    # Group patches once to avoid repeatedly filtering.
    patch_groups_r1 = {pid: g for pid, g in df_r1.groupby("patch_id")}
    patch_groups_r2 = {pid: g for pid, g in df_r2.groupby("patch_id")}

    rows: list[dict] = []
    for _, cand in candidate_matches.iterrows():
        cid1 = cand[id_r1_col]
        cid2 = cand[id_r2_col]
        if cid1 not in r1_lookup.index or cid2 not in r2_lookup.index:
            continue
        cell1 = r1_lookup.loc[cid1]
        cell2 = r2_lookup.loc[cid2]
        if (cell1["patch_x"], cell1["patch_y"]) != (cell2["patch_x"], cell2["patch_y"]):
            continue  # must be in the same patch
        patch_id = cell1["patch_id"]
        df_r1_patch = patch_groups_r1.get(patch_id)
        df_r2_patch = patch_groups_r2.get(patch_id)
        if df_r1_patch is None or df_r2_patch is None:
            continue

        l_pos = compute_L_pos(cell1, cell2, image_width, image_height)
        if l_pos > tau_pos:
            continue
        l_nei = compute_L_nei(
            cell1,
            cell2,
            df_r1_patch,
            df_r2_patch,
            k,
            image_width,
            image_height,
            x_col=x_col,
            y_col=y_col,
        )
        if l_nei > tau_nei:
            continue

        rows.append(
            {
                id_r1_col: cid1,
                id_r2_col: cid2,
                "L_pos": l_pos,
                "L_nei": l_nei,
                "patch_x": cell1["patch_x"],
                "patch_y": cell1["patch_y"],
            }
        )

    return pd.DataFrame(rows)


def map_neighbors_for_pair(
    pair: pd.Series,
    df_r1_patch: pd.DataFrame,
    df_r2_patch: pd.DataFrame,
    k_neighbor: int,
    tau_map: float,
    image_width: float,
    image_height: float,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> list[dict]:
    """Map neighbors around a trusted pair using relative displacement."""
    if df_r1_patch.empty or df_r2_patch.empty:
        return []
    anchor1 = df_r1_patch[df_r1_patch[cell_id_col] == pair[id_r1_col]]
    anchor2 = df_r2_patch[df_r2_patch[cell_id_col] == pair[id_r2_col]]
    if anchor1.empty or anchor2.empty:
        return []
    anchor1 = anchor1.iloc[0]
    anchor2 = anchor2.iloc[0]
    diag = _patch_diag_px(image_width, image_height)
    rows: list[dict] = []

    coords_r1 = df_r1_patch[[x_col, y_col]].to_numpy()
    nn1 = NearestNeighbors(
        n_neighbors=min(k_neighbor + 1, len(df_r1_patch)),
        algorithm="kd_tree",
    ).fit(coords_r1)
    idxs1 = nn1.kneighbors([[anchor1[x_col], anchor1[y_col]]], return_distance=False)
    coords_r2 = df_r2_patch[[x_col, y_col]].to_numpy()
    nn2 = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords_r2)

    for idx in idxs1[0][1:]:  # skip self (index 0)
        neigh = df_r1_patch.iloc[idx]
        dx = neigh[x_col] - anchor1[x_col]
        dy = neigh[y_col] - anchor1[y_col]
        x_pred = anchor2[x_col] + dx
        y_pred = anchor2[y_col] + dy
        pred_query = np.array([[x_pred, y_pred]], dtype=float)
        dist_px, idx2 = nn2.kneighbors(pred_query, return_distance=True)
        dist_norm = float(dist_px[0][0] / diag)
        target = df_r2_patch.iloc[idx2[0][0]]
        rows.append(
            {
                "cell_id_r1": neigh[cell_id_col],
                "cell_id_r2": target[cell_id_col],
                "pair_top_r1": anchor1[cell_id_col],
                "pair_top_r2": anchor2[cell_id_col],
                "dist_norm": dist_norm,
                "within_threshold": dist_norm <= tau_map,
                "patch_x": pair["patch_x"],
                "patch_y": pair["patch_y"],
            }
        )
    return rows


def map_all_neighbors(
    top_pairs: pd.DataFrame,
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    k_neighbor: int,
    tau_map: float,
    image_width: float,
    image_height: float,
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Run neighbor mapping for all trusted top pairs."""
    all_rows: list[dict] = []
    for _, pair in top_pairs.iterrows():
        px, py = int(pair["patch_x"]), int(pair["patch_y"])
        df_r1_patch = df_r1[(df_r1["patch_x"] == px) & (df_r1["patch_y"] == py)]
        df_r2_patch = df_r2[(df_r2["patch_x"] == px) & (df_r2["patch_y"] == py)]
        mapped = map_neighbors_for_pair(
            pair,
            df_r1_patch,
            df_r2_patch,
            k_neighbor,
            tau_map,
            image_width,
            image_height,
            cell_id_col=cell_id_col,
        )
        all_rows.extend(mapped)
    if not all_rows:
        return pd.DataFrame(
            columns=[
                "cell_id_r1",
                "cell_id_r2",
                "pair_top_r1",
                "pair_top_r2",
                "dist_norm",
                "within_threshold",
                "patch_x",
        "patch_y",
    ]
    )
    return pd.DataFrame(all_rows)


def run_topology_matching_df(
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    candidate_matches: pd.DataFrame,
    image_width: float,
    image_height: float,
    k_pos_nei: int,
    k_neighbor: int,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    tau_map: float = 0.3,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    In-memory version of the topology-based matching pipeline.
    """
    trusted_pairs = filter_top_pairs(
        candidate_matches,
        df_r1,
        df_r2,
        k=k_pos_nei,
        image_width=image_width,
        image_height=image_height,
        tau_pos=tau_pos,
        tau_nei=tau_nei,
        id_r1_col=id_r1_col,
        id_r2_col=id_r2_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
    )
    neighbor_matches = map_all_neighbors(
        trusted_pairs,
        df_r1,
        df_r2,
        k_neighbor=k_neighbor,
        tau_map=tau_map,
        image_width=image_width,
        image_height=image_height,
        cell_id_col=cell_id_col,
    )
    return trusted_pairs, neighbor_matches


def run_topology_matching(
    round1_csv: Path | str,
    round2_csv: Path | str,
    candidate_csv: Path | str,
    image_width: float,
    image_height: float,
    k_pos_nei: int,
    k_neighbor: int,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    tau_map: float = 0.3,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline entrypoint as described in the prompt."""
    df_r1 = load_cells(round1_csv, image_width, image_height, id_col=cell_id_col, x_col=x_col, y_col=y_col)
    df_r2 = load_cells(round2_csv, image_width, image_height, id_col=cell_id_col, x_col=x_col, y_col=y_col)
    candidate_matches = pd.read_csv(candidate_csv)
    return run_topology_matching_df(
        df_r1,
        df_r2,
        candidate_matches,
        image_width=image_width,
        image_height=image_height,
        k_pos_nei=k_pos_nei,
        k_neighbor=k_neighbor,
        tau_pos=tau_pos,
        tau_nei=tau_nei,
        tau_map=tau_map,
        id_r1_col=id_r1_col,
        id_r2_col=id_r2_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
    )


def compute_match_residuals(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
    transform: RigidTransform,
) -> np.ndarray:
    """Compute Euclidean residuals (in pixels) for each match after applying transform to round2."""
    if matches.empty:
        return np.array([], dtype=float)
    pts1 = feats1.iloc[matches["idx1"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[matches["idx2"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2_warp = (transform.rotation @ pts2.T).T + transform.translation
    return np.linalg.norm(pts1 - pts2_warp, axis=1)

def run_pipeline(
    img1_path: Path = DEFAULT_IMG1_PATH,
    img2_path: Path = DEFAULT_IMG2_PATH,
    top_k: int = DEFAULT_TOP_K,
    top_k_per_patch: int | None = TOP_K_PER_PATCH,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    napari_view: bool = False,
    segmentation_only: bool = False,
    save_match_table: Path = DEFAULT_SAVE_MATCH_TABLE,
    save_match_overlay_path: Path = DEFAULT_SAVE_MATCH_OVERLAY,
    save_segmentation_prefix: Path = DEFAULT_SAVE_SEGMENTATION_PREFIX,
    save_match_plot_path: Path = DEFAULT_SAVE_MATCH_PLOT,
    save_registration_overlay_path: Path = DEFAULT_SAVE_REGISTRATION_OVERLAY,
    save_features_dir: Path | None = DEFAULT_SAVE_FEATURES_DIR,
    residual_prune_quantile: float | None = DEFAULT_RESIDUAL_PRUNE_QUANTILE,
    use_spatial_clusters: bool = False,
    n_clusters: int = 9,
    use_topology_filtering: bool = False,
    k_pos_nei: int = 5,
    k_neighbor: int = 5,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    tau_map: float = 0.3,
    use_ransac_transform: bool = False,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    save_npy: bool = False,
) -> None:
    segmenter = CellposeSegmenter(DEFAULT_CELLPOSE_CONFIG)

    # Detect single-image mode (when same path is provided for both images)
    single_image_mode = (img1_path == img2_path) and segmentation_only
    
    img1 = load_image(img1_path)
    
    if single_image_mode:
        # Only process one image in single-image segmentation mode
        mode1 = infer_image_mode(img1)
        use_zstack = mode1 == "3d_zstack"
        
        if use_zstack:
            _masks1_3d, masks1, _, _ = segmenter.segment_zstack(img1)
            overlay_img1 = project_intensity_max(img1)
        else:
            masks1, _, _ = segmenter.segment_array(img1)
            overlay_img1 = img1
        
        # Save and/or display the single segmentation
        if save_segmentation_prefix is not None:
            import imageio.v3 as iio
            
            out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + ".tif")
            iio.imwrite(out1, masks1.astype(np.uint16))
            print(f"Saved segmentation mask to {out1}")

            if save_npy:
                npy_out = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + ".npy")
                np.save(npy_out, masks1)
                print(f"Saved segmentation npy to {npy_out}")
        
        if napari_view:
            try:
                import napari  # type: ignore
                from .visualization import launch_napari_viewer
            except ImportError:
                print("napari not installed; skipping interactive viewer.")
            else:
                viewer = launch_napari_viewer(overlay_img1, masks1, title=f"{img1_path.name} segmentation")
                viewer.window._qt_window.raise_()
                napari.run()
        return
    
    # Two-image mode (original behavior)
    img2 = load_image(img2_path)

    mode1 = infer_image_mode(img1)
    mode2 = infer_image_mode(img2)
    if mode1 != mode2:
        raise ValueError(f"Both images must be either 2D or 3D Z-stacks; got {mode1} and {mode2}.")

    use_zstack = mode1 == "3d_zstack"

    if use_zstack:
        _masks1_3d, masks1, _, _ = segmenter.segment_zstack(img1)
        _masks2_3d, masks2, _, _ = segmenter.segment_zstack(img2)
        overlay_img1 = project_intensity_max(img1)
        overlay_img2 = project_intensity_max(img2)
    else:
        masks1, _, _ = segmenter.segment_array(img1)
        masks2, _, _ = segmenter.segment_array(img2)    
        overlay_img1 = img1
        overlay_img2 = img2

    if segmentation_only:
        if save_segmentation_prefix is not None:
            import imageio.v3 as iio

            out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img1.tif")
            out2 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img2.tif")
            iio.imwrite(out1, masks1.astype(np.uint16))
            iio.imwrite(out2, masks2.astype(np.uint16))
            print(f"Saved segmentation masks to {out1} and {out2}")

            if save_npy:
                npy_out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img1.npy")
                npy_out2 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img2.npy")
                np.save(npy_out1, masks1)
                np.save(npy_out2, masks2)
                print(f"Saved segmentation npy to {npy_out1} and {npy_out2}")

        if napari_view:
            try:
                import napari  # type: ignore
                from .visualization import launch_napari_viewer
            except ImportError:
                print("napari not installed; skipping interactive viewer.")
            else:
                viewer1 = launch_napari_viewer(overlay_img1, masks1, title="Image1 segmentation")
                viewer2 = launch_napari_viewer(overlay_img2, masks2, title="Image2 segmentation")
                viewer1.window._qt_window.raise_()
                viewer2.window._qt_window.raise_()
                napari.run()
        return

    feats1 = compute_cell_features(masks1, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
    feats2 = compute_cell_features(masks2, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
    h1, w1 = masks1.shape
    h2, w2 = masks2.shape

    # --- Global Registration Step ---
    print("Running robust global alignment...")
    global_transform = perform_global_registration(feats1, feats2)

    if global_transform is not None:
        rot = getattr(global_transform, "rotation", np.nan)
        trans = getattr(global_transform, "translation", np.array([np.nan, np.nan]))
        if not (np.isfinite(rot) and np.all(np.isfinite(trans))):
            print("Global alignment returned non-finite parameters; skipping global transform.")
            global_transform = None

    if global_transform is not None:
        print("Global alignment successful.")
        print(f"Initial rotation: {global_transform.rotation}")
        print(f"Initial translation: {global_transform.translation}")
        # Apply transform to create a temporary aligned dataframe for matching
        feats2_aligned = apply_transform_to_coordinates(feats2, global_transform)
    else:
        print("Global alignment failed or insufficient matches. Proceeding with raw coordinates.")
        feats2_aligned = feats2.copy()

    feats1 = assign_patches(feats1, w1, h1, x_col="centroid_x", y_col="centroid_y")
    feats2_aligned = assign_patches(feats2_aligned, w1, h1, x_col="centroid_x", y_col="centroid_y")

    if use_spatial_clusters:
        feats1, feats2_aligned = assign_clusters_from_round1(
            feats1,
            feats2_aligned,
            n_clusters=n_clusters,
            x_col="centroid_x",
            y_col="centroid_y",
            cluster_col="cluster_id",
        )

    if save_features_dir is not None:
        save_features_dir.mkdir(parents=True, exist_ok=True)
        # Add convenience x/y aliases expected by point_registration
        feats1_to_save = feats1.copy()
        feats2_to_save = feats2.copy()
        feats1_to_save["x"] = feats1_to_save["centroid_x"]
        feats1_to_save["y"] = feats1_to_save["centroid_y"]
        feats2_to_save["x"] = feats2_to_save["centroid_x"]
        feats2_to_save["y"] = feats2_to_save["centroid_y"]
        out1 = save_features_dir / "round1_cells.csv"
        out2 = save_features_dir / "round2_cells.csv"
        feats1_to_save.to_csv(out1, index=False)
        feats2_to_save.to_csv(out2, index=False)
        print(f"Saved feature tables to {out1} and {out2}")

    match_cfg = MatchingConfig(top_k=top_k, position_weight=position_weight)

    if use_spatial_clusters:
        matches = match_cells_per_cluster(
            feats1, feats2_aligned, match_cfg, cluster_col="cluster_id", top_k_per_cluster=top_k_per_patch
        )
    else:
        if top_k_per_patch is not None:
            matches = match_cells_per_patch(feats1, feats2_aligned, match_cfg, top_k_per_patch=top_k_per_patch)
        else:
            matches = greedy_match_cells(feats1, feats2_aligned, match_cfg)

    if use_topology_filtering:
        if "cell_id" not in feats1.columns:
            feats1 = feats1.copy()
            feats1["cell_id"] = feats1.index
        if "cell_id" not in feats2.columns:
            feats2 = feats2.copy()
            feats2["cell_id"] = feats2.index

        candidate_matches = pd.DataFrame(
            {
                "cell_id_r1": feats1.loc[matches["idx1"], "cell_id"].to_numpy() if not matches.empty else [],
                "cell_id_r2": feats2_aligned.loc[matches["idx2"], "cell_id"].to_numpy() if not matches.empty else [],
            }
        )

        trusted_pairs, neighbor_matches = run_topology_matching_df(
            feats1,
            feats2_aligned,
            candidate_matches,
            image_width=w1,
            image_height=h1,
            k_pos_nei=k_pos_nei,
            k_neighbor=k_neighbor,
            tau_pos=tau_pos,
            tau_nei=tau_nei,
            tau_map=tau_map,
            id_r1_col="cell_id_r1",
            id_r2_col="cell_id_r2",
            cell_id_col="cell_id",
            x_col="centroid_x",
            y_col="centroid_y",
        )

        combined_pairs = trusted_pairs.copy()
        neighbor_filtered = neighbor_matches
        if not neighbor_matches.empty and "within_threshold" in neighbor_matches.columns:
            neighbor_filtered = neighbor_matches[neighbor_matches["within_threshold"] == True]
        if not neighbor_filtered.empty:
            combined_pairs = pd.concat([combined_pairs, neighbor_filtered], ignore_index=True)

        if combined_pairs.empty:
            matches = pd.DataFrame(columns=["idx1", "idx2", "cell_id_1", "cell_id_2"])
        else:
            combined_pairs = combined_pairs.drop_duplicates(subset=["cell_id_r1", "cell_id_r2"])
            id_to_idx1 = {feats1.loc[i, "cell_id"]: i for i in feats1.index}
            id_to_idx2 = {feats2_aligned.loc[i, "cell_id"]: i for i in feats2_aligned.index}
            rows = []
            for _, pair in combined_pairs.iterrows():
                cid1 = pair["cell_id_r1"]
                cid2 = pair["cell_id_r2"]
                idx1 = id_to_idx1.get(cid1)
                idx2 = id_to_idx2.get(cid2)
                if idx1 is None or idx2 is None:
                    continue
                row = {"idx1": idx1, "idx2": idx2, "cell_id_1": cid1, "cell_id_2": cid2}
                for extra in ("L_pos", "L_nei", "dist_norm", "within_threshold", "patch_x", "patch_y"):
                    if extra in pair and not pd.isna(pair[extra]):
                        row[extra] = pair[extra]
                rows.append(row)
            matches = pd.DataFrame(rows)
        matches = matches.reset_index(drop=True)

    print("Top matches:")
    print(matches.head(top_k))

    if matches.empty:
        print("No matches available after matching; aborting registration.")
        return

    if use_ransac_transform:
        transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1,
            feats2,
            matches,
            max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches["residual_px"] = residuals
    else:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches, use_scale=True)
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches["residual_px"] = residuals

    # Prune high-residual matches and re-estimate transform for tighter alignment.
    if (
        residual_prune_quantile is not None
        and 0.0 < residual_prune_quantile < 1.0
        and len(residuals) >= MIN_MATCHES_FOR_REFINEMENT
    ):
        threshold = float(np.quantile(residuals, residual_prune_quantile))
        keep_mask = residuals <= threshold
        kept = int(keep_mask.sum())
        if kept >= MIN_MATCHES_FOR_REFINEMENT and kept < len(matches):
            matches = matches.loc[keep_mask].reset_index(drop=True)
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches, use_scale=True)
            refined_residuals = compute_match_residuals(feats1, feats2, matches, transform)
            matches["residual_px"] = refined_residuals
            print(
                f"Pruned high-residual matches (>{threshold:.3f}px); kept {kept} / {len(residuals)} and recomputed transform."
            )
        else:
            print("Residual pruning skipped (not enough inliers to refine).")
    else:
        print("Residual pruning disabled or insufficient matches to apply.")

    print("Estimated rotation matrix:")
    print(transform.rotation)
    print("Estimated translation vector:")
    print(transform.translation)

    if save_match_table is not None:
        from .visualization import export_match_table

        export_match_table(matches, save_match_table)
        print(f"Saved match table to {save_match_table}")

    if save_segmentation_prefix is not None:
        from .visualization import save_segmentation_plot

        out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img1.tif")
        out2 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img2.tif")
        save_segmentation_plot(overlay_img1, masks1, out1, title="Segmentation image1")
        save_segmentation_plot(overlay_img2, masks2, out2, title="Segmentation image2")
        print(f"Saved segmentation plots to {out1} and {out2}")

    if save_match_overlay_path is not None:
        from .visualization import save_match_overlay

        overlay_path = save_match_overlay_path.with_suffix(".tif")
        save_match_overlay(overlay_img1, overlay_img2, feats1, feats2, matches, overlay_path)
        print(f"Saved match overlay image to {overlay_path}")

    if save_match_plot_path is not None:
        from .visualization import save_match_plot

        match_plot_path = save_match_plot_path.with_suffix(".tif")
        save_match_plot(overlay_img1, overlay_img2, feats1, feats2, matches, match_plot_path)
        print(f"Saved match plot to {match_plot_path}")

    if save_registration_overlay_path is not None:
        from .visualization import save_registration_overlay

        reg_overlay_path = save_registration_overlay_path.with_suffix(".tif")
        save_registration_overlay(
            overlay_img1,
            masks1,
            overlay_img2,
            rotation=transform.rotation,
            translation=transform.translation,
            path=reg_overlay_path,
        )
        print(f"Saved registration overlay to {reg_overlay_path}")

    if napari_view:
        try:
            import napari  # type: ignore
        except ImportError:
            print("napari not installed; skipping interactive viewer.")
        else:
            if top_k_per_patch is not None:
                if matches.empty:
                    print("No top-per-region matches to display in napari.")
                else:
                    pts_r1 = feats1.loc[matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
                    pts_r2 = feats2.loc[matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
                    pts_r2_reg = _apply_rigid_to_points(pts_r2, transform.rotation, transform.translation)
                    title = "Top-per-cluster registration" if use_spatial_clusters else "Top-per-patch registration (54 cells)"
                    viewer = napari.Viewer(title=title)
                    viewer.add_points(pts_r1, name="round1 top", face_color="cyan", size=12, opacity=0.9)
                    viewer.add_points(pts_r2, name="round2 top (pre)", face_color="orange", size=10, opacity=0.6)
                    viewer.add_points(pts_r2_reg, name="round2 top (reg)", face_color="magenta", size=12, opacity=0.9)
                    viewer.window._qt_window.raise_()
                    napari.run()
            else:
                try:
                    from .visualization import launch_napari_viewer, warp_mask_to_image2
                except ImportError:
                    print("napari not installed; skipping interactive viewer.")
                else:
                    viewer1 = launch_napari_viewer(overlay_img1, masks1, feats1, title="Image1")
                    viewer2 = launch_napari_viewer(overlay_img2, masks2, feats2, title="Image2")
                    warped = warp_mask_to_image2(
                        masks1, overlay_img2.shape[:2], transform.rotation, transform.translation
                    )
                    viewer2.add_labels((warped > 0).astype(int), name="warped_mask1_on_image2", opacity=0.4)
                    viewer1.window._qt_window.raise_()  # bring windows forward
                    viewer2.window._qt_window.raise_()
                    napari.run()
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell registration demo using Cellpose-SAM.")
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="Path to image(s). For registration, provide exactly 2 images. For --segmentation-only, provide any number of images.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of matches to use.")
    parser.add_argument(
        "--position-weight",
        type=float,
        default=DEFAULT_POSITION_WEIGHT,
        help="Weight for spatial proximity in matching (0 to disable position).",
    )
    parser.add_argument(
        "--top-per-patch",
        type=int,
        default=TOP_K_PER_PATCH,
        help="If set, pick this many best matches per 3x3 patch (e.g., 6 -> up to 54 total).",
    )
    parser.add_argument(
        "--use-spatial-clusters",
        action="store_true",
        help="Use adaptive spatial clustering instead of fixed 3x3 patches for matching.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=9,
        help="Number of spatial clusters when --use-spatial-clusters is enabled.",
    )
    parser.add_argument(
        "--use-topology-filtering",
        action="store_true",
        help="Enable topology-based filtering/mapping of candidate matches.",
    )
    parser.add_argument("--k-pos-nei", type=int, default=5, help="k for position/neighborhood filtering.")
    parser.add_argument("--k-neighbor", type=int, default=5, help="k for neighbor mapping.")
    parser.add_argument(
        "--tau-pos",
        type=float,
        default=0.5,
        help="Threshold for positional similarity during topology filtering.",
    )
    parser.add_argument(
        "--tau-nei",
        type=float,
        default=0.3,
        help="Threshold for neighbor-structure similarity during topology filtering.",
    )
    parser.add_argument(
        "--tau-map",
        type=float,
        default=0.3,
        help="Threshold on mapped neighbor distance (normalized) during topology filtering.",
    )
    parser.add_argument(
        "--use-ransac-transform",
        action="store_true",
        help="Estimate the rigid transform with a RANSAC wrapper for robustness.",
    )
    parser.add_argument(
        "--ransac-max-trials",
        type=int,
        default=1000,
        help="Maximum RANSAC iterations for robust transform estimation.",
    )
    parser.add_argument(
        "--ransac-residual-threshold",
        type=float,
        default=2.0,
        help="Residual threshold (pixels) to count an inlier in RANSAC.",
    )
    parser.add_argument("--napari", action="store_true", help="Open napari viewers for segmentation results.")
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Run segmentation (and optional napari view) only; skip feature extraction, matching, registration.",
    )
    parser.add_argument(
        "--save-match-table",
        type=Path,
        default=DEFAULT_SAVE_MATCH_TABLE,
        help="Path to save CSV of matches.",
    )
    parser.add_argument(
        "--save-match-overlay",
        type=Path,
        default=DEFAULT_SAVE_MATCH_OVERLAY,
        help="Path to save overlay of matched centroids (forced TIF).",
    )
    parser.add_argument(
        "--save-segmentation-prefix",
        type=Path,
        default=DEFAULT_SAVE_SEGMENTATION_PREFIX,
        help="Path prefix for saving segmentation plots (saves *_img1.tif and *_img2.tif).",
    )
    parser.add_argument(
        "--save-match-plot",
        type=Path,
        default=DEFAULT_SAVE_MATCH_PLOT,
        help="Path to save matplotlib match plot (forced TIF).",
    )
    parser.add_argument(
        "--save-registration-overlay",
        type=Path,
        default=DEFAULT_SAVE_REGISTRATION_OVERLAY,
        help="Path to save registration overlay (mask1 warped onto image2, TIF).",
    )
    parser.add_argument(
        "--save-features-dir",
        type=Path,
        default=DEFAULT_SAVE_FEATURES_DIR,
        help="Directory to save per-cell feature tables (round1_cells.csv, round2_cells.csv).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save segmentation masks as .npy files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle batch segmentation mode
    if args.segmentation_only:
        if len(args.images) == 1:
            # Single image segmentation
            print(f"Processing single image: {args.images[0]}")
            run_pipeline(
                args.images[0],
                args.images[0],  # Dummy second image, won't be used in segmentation_only mode
                napari_view=args.napari,
                segmentation_only=True,
                save_segmentation_prefix=args.save_segmentation_prefix,
                save_npy=args.save_npy,
            )
        else:
            # Batch segmentation for multiple images
            print(f"Processing {len(args.images)} images in batch mode...")
            for i, img_path in enumerate(args.images, 1):
                print(f"\n[{i}/{len(args.images)}] Processing: {img_path}")
                # Create unique output prefix for each image
                if args.save_segmentation_prefix:
                    prefix = args.save_segmentation_prefix.with_name(
                        f"{args.save_segmentation_prefix.stem}_image{i}"
                    )
                else:
                    prefix = None
                
                run_pipeline(
                    img_path,
                    img_path,  # Dummy second image
                    napari_view=False,  # Disable napari in batch mode
                    segmentation_only=True,
                    save_segmentation_prefix=prefix,
                    save_npy=args.save_npy,
                )
            
            # Optionally open napari after all processing if requested
            if args.napari:
                print("\nBatch processing complete. Napari viewing not supported for batch mode.")
                print("Please view individual segmentation outputs in the outputs directory.")
    else:
        # Registration mode requires exactly 2 images
        if len(args.images) != 2:
            print(f"Error: Registration mode requires exactly 2 images, but {len(args.images)} were provided.")
            print("For batch segmentation of multiple images, use --segmentation-only flag.")
            sys.exit(1)
        
        run_pipeline(
            args.images[0],
            args.images[1],
            top_k=args.top_k,
            top_k_per_patch=args.top_per_patch,
            position_weight=args.position_weight,
            napari_view=args.napari,
            segmentation_only=args.segmentation_only,
            save_match_table=args.save_match_table,
            save_match_overlay_path=args.save_match_overlay,
            save_segmentation_prefix=args.save_segmentation_prefix,
            save_match_plot_path=args.save_match_plot,
            save_registration_overlay_path=args.save_registration_overlay,
            save_features_dir=args.save_features_dir,
            use_spatial_clusters=args.use_spatial_clusters,
            n_clusters=args.n_clusters,
            use_topology_filtering=args.use_topology_filtering,
            k_pos_nei=args.k_pos_nei,
            k_neighbor=args.k_neighbor,
            tau_pos=args.tau_pos,
            tau_nei=args.tau_nei,
            tau_map=args.tau_map,
            use_ransac_transform=args.use_ransac_transform,
            ransac_max_trials=args.ransac_max_trials,
            ransac_residual_threshold=args.ransac_residual_threshold,
            save_npy=args.save_npy,
        )


if __name__ == "__main__":
    main()



