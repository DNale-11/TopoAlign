from __future__ import annotations
import argparse
import json
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Literal
import numpy as np
import pandas as pd
import imageio.v3 as iio
import tifffile as tiff
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.transform import AffineTransform, warp
from .config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .matching import (
    MatchingConfig,
    match_cells_per_cluster,
    two_stage_match_cells,
)
from .point_registration import estimate_robust_transform
from .registration import RigidTransform, estimate_rigid_transform_from_matches
try:
    from .segmentation import CellposeSegmenter
except ImportError:  # Cellpose is optional for help and precomputed inputs.
    CellposeSegmenter = None
from .io_utils import infer_image_mode, load_image, project_intensity_max

# Allow running as `python cell_registration/main.py` by setting package context.
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "cell_registration"

# Single source of truth for defaults.
DEFAULT_IMG1_PATH = Path("1_ch5.tif")
DEFAULT_IMG2_PATH = Path("2_ch5.tif")
DEFAULT_TOP_K = 50
DEFAULT_FEATURE_WEIGHT = 1.0
DEFAULT_TOPOLOGY_WEIGHT = 0.35
DEFAULT_POSITION_WEIGHT = 4.0
DEFAULT_DISTANCE_THRESHOLD = 2.0
DEFAULT_SPATIAL_WINDOW_SIZE = 100.0
DEFAULT_SAVE_MATCH_TABLE = Path("outputs/top_matches.csv")
DEFAULT_SAVE_MATCH_OVERLAY = Path("outputs/match _overlay")
DEFAULT_SAVE_SEGMENTATION_PREFIX = Path("outputs/segmentation")
DEFAULT_SAVE_MATCH_PLOT = Path("outputs/match_plot")
DEFAULT_SAVE_REGISTRATION_OVERLAY = Path("outputs/registration_overlay")
DEFAULT_SAVE_REGISTERED_MOVING = None
DEFAULT_SAVE_FEATURES_DIR = Path("outputs")
DEFAULT_RESIDUAL_PRUNE_QUANTILE = None
MIN_MATCHES_FOR_REFINEMENT = 3

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


ChannelAxis = Literal["auto", "first", "last", "none"]


def _resolve_channel_index(channel_count: int, channel: int) -> int:
    idx = int(channel)
    if idx < 0:
        idx += channel_count
    if idx < 0 or idx >= channel_count:
        raise ValueError(f"registration_channel {channel} is out of range for {channel_count} channels.")
    return idx


def _extract_registration_channel(
    img: np.ndarray,
    channel_axis: ChannelAxis,
    registration_channel: int,
) -> np.ndarray:
    """
    Extract the 2D intensity image used for segmentation and landmark registration.

    The explicit channel-axis modes are for multi-channel 2D images. The "auto" mode
    preserves the previous pipeline behavior.
    """
    if channel_axis == "none":
        if img.ndim != 2:
            raise ValueError(f"--channel-axis none expects a 2D image, got shape {img.shape}.")
        return img

    if channel_axis == "first":
        if img.ndim != 3:
            raise ValueError(f"--channel-axis first expects a C,Y,X image, got shape {img.shape}.")
        ch = _resolve_channel_index(img.shape[0], registration_channel)
        return img[ch, :, :]

    if channel_axis == "last":
        if img.ndim != 3:
            raise ValueError(f"--channel-axis last expects a Y,X,C image, got shape {img.shape}.")
        ch = _resolve_channel_index(img.shape[-1], registration_channel)
        return img[..., ch]

    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] <= 4:
        ch = _resolve_channel_index(img.shape[-1], registration_channel)
        return img[..., ch]
    raise ValueError(
        f"Cannot infer a 2D registration channel from shape {img.shape}; "
        "use --channel-axis first or --channel-axis last for multi-channel images."
    )


def _rigid_transform_to_affine(transform: RigidTransform) -> AffineTransform:
    R = np.asarray(transform.rotation, dtype=float)
    t = np.asarray(transform.translation, dtype=float)
    return AffineTransform(
        matrix=np.array(
            [
                [R[0, 0], R[0, 1], t[0]],
                [R[1, 0], R[1, 1], t[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )


def _load_affine_transform_json(path: Path | str | None) -> np.ndarray | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in (
            "matrix",
            "affine",
            "transform",
            "coarse_affine",
            "global_affine",
            "global_affine_moving_to_fixed",
            "coarse_affine_moving_to_fixed",
        ):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                f"No affine matrix found in {path}; expected one of "
                "matrix/affine/transform/coarse_affine/global_affine_moving_to_fixed."
            )
    matrix = np.asarray(data, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Affine transform JSON must contain a 3x3 matrix, got {matrix.shape}.")
    return matrix


def _cast_warped_like_input(warped: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(warped, info.min, info.max)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return warped.astype(dtype)
    return warped


def warp_moving_image_to_fixed(
    moving_image: np.ndarray,
    transform: RigidTransform,
    output_shape: tuple[int, int],
    channel_axis: ChannelAxis,
    order: int = 1,
) -> np.ndarray:
    """
    Apply a moving->fixed transform to a 2D or multi-channel moving image.

    For C,Y,X input with --channel-axis first, the output keeps C,Y,X layout.
    """
    affine = _rigid_transform_to_affine(transform)
    source_dtype = np.asarray(moving_image).dtype

    def warp_plane(plane: np.ndarray) -> np.ndarray:
        return warp(
            plane.astype(float),
            inverse_map=affine.inverse,
            output_shape=output_shape,
            preserve_range=True,
            order=order,
            mode="constant",
            cval=0.0,
        )

    if channel_axis == "first":
        if moving_image.ndim != 3:
            raise ValueError(f"--channel-axis first expects a C,Y,X moving image, got shape {moving_image.shape}.")
        warped = np.stack([warp_plane(moving_image[c]) for c in range(moving_image.shape[0])], axis=0)
        return _cast_warped_like_input(warped, source_dtype)

    if channel_axis == "last":
        if moving_image.ndim != 3:
            raise ValueError(f"--channel-axis last expects a Y,X,C moving image, got shape {moving_image.shape}.")
        warped = np.stack([warp_plane(moving_image[..., c]) for c in range(moving_image.shape[-1])], axis=-1)
        return _cast_warped_like_input(warped, source_dtype)

    if channel_axis == "none" or moving_image.ndim == 2:
        return _cast_warped_like_input(warp_plane(moving_image), source_dtype)

    if channel_axis == "auto" and moving_image.ndim == 3 and moving_image.shape[-1] <= 4:
        warped = np.stack([warp_plane(moving_image[..., c]) for c in range(moving_image.shape[-1])], axis=-1)
        return _cast_warped_like_input(warped, source_dtype)

    raise ValueError(
        f"Cannot infer how to warp moving image shape {moving_image.shape}; "
        "use --channel-axis first or --channel-axis last."
    )


def _axes_metadata_for_registered_image(img: np.ndarray, channel_axis: ChannelAxis) -> str | None:
    if img.ndim == 2:
        return "YX"
    if img.ndim == 3 and channel_axis == "first":
        return "CYX"
    if img.ndim == 3 and channel_axis in ("last", "auto"):
        return "YXC"
    return None


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


def enforce_one_to_one_pairs(
    pairs: pd.DataFrame,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
) -> pd.DataFrame:
    """
    Keep only a global 1-to-1 subset of candidate pairs.

    Trusted anchor pairs are preferred over propagated neighbor pairs, and within
    each stage lower scores are preferred.
    """
    if pairs.empty:
        return pairs.copy()

    out = pairs.copy()
    if "pair_stage" not in out.columns:
        out["pair_stage"] = 0
    if "pair_score" not in out.columns:
        out["pair_score"] = np.inf

    out["pair_stage"] = out["pair_stage"].fillna(1).astype(int)
    out["pair_score"] = pd.to_numeric(out["pair_score"], errors="coerce").fillna(np.inf)
    out = out.drop_duplicates(subset=[id_r1_col, id_r2_col])
    out = out.sort_values(
        by=["pair_stage", "pair_score", id_r1_col, id_r2_col],
        ascending=[True, True, True, True],
    )

    used_r1: set = set()
    used_r2: set = set()
    kept_rows: list[pd.Series] = []
    for _, row in out.iterrows():
        cid1 = row[id_r1_col]
        cid2 = row[id_r2_col]
        if cid1 in used_r1 or cid2 in used_r2:
            continue
        kept_rows.append(row)
        used_r1.add(cid1)
        used_r2.add(cid2)

    if not kept_rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def rebuild_matches_from_topology_pairs(
    base_matches: pd.DataFrame,
    fixed_feats: pd.DataFrame,
    moving_feats: pd.DataFrame,
    *,
    image_width: float,
    image_height: float,
    k_pos_nei: int,
    k_neighbor: int,
    tau_pos: float,
    tau_nei: float,
    tau_map: float,
) -> pd.DataFrame:
    """
    Rebuild the final match table from topology-filtered pairs.

    Trusted anchor pairs are preferred over propagated neighbors and the final
    result is forced to be a global 1-to-1 matching.
    """
    if base_matches.empty:
        return pd.DataFrame(columns=["idx1", "idx2", "cell_id_1", "cell_id_2"])

    candidate_matches = pd.DataFrame(
        {
            "cell_id_r1": fixed_feats.loc[base_matches["idx1"], "cell_id"].to_numpy(),
            "cell_id_r2": moving_feats.loc[base_matches["idx2"], "cell_id"].to_numpy(),
        }
    )
    trusted_pairs, neighbor_matches = run_topology_matching_df(
        fixed_feats,
        moving_feats,
        candidate_matches,
        image_width=image_width,
        image_height=image_height,
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
    if not combined_pairs.empty:
        combined_pairs["pair_stage"] = 0
        combined_pairs["pair_score"] = (
            pd.to_numeric(combined_pairs.get("L_pos"), errors="coerce").fillna(np.inf)
            + pd.to_numeric(combined_pairs.get("L_nei"), errors="coerce").fillna(np.inf)
        )

    neighbor_filtered = neighbor_matches
    if not neighbor_matches.empty and "within_threshold" in neighbor_matches.columns:
        neighbor_filtered = neighbor_matches[neighbor_matches["within_threshold"] == True]
    if not neighbor_filtered.empty:
        neighbor_filtered = neighbor_filtered.copy()
        neighbor_filtered["pair_stage"] = 1
        neighbor_filtered["pair_score"] = pd.to_numeric(
            neighbor_filtered.get("dist_norm"),
            errors="coerce",
        ).fillna(np.inf)
        combined_pairs = pd.concat([combined_pairs, neighbor_filtered], ignore_index=True)

    if combined_pairs.empty:
        print("Topology filtering removed all candidate matches.")
        return pd.DataFrame(columns=["idx1", "idx2", "cell_id_1", "cell_id_2"])

    combined_pairs = enforce_one_to_one_pairs(combined_pairs)
    id_to_idx1 = {fixed_feats.loc[idx, "cell_id"]: idx for idx in fixed_feats.index}
    id_to_idx2 = {moving_feats.loc[idx, "cell_id"]: idx for idx in moving_feats.index}

    base_lookup: dict[tuple[object, object], pd.Series] = {}
    for _, base_row in base_matches.iterrows():
        key = (
            fixed_feats.loc[base_row["idx1"], "cell_id"],
            moving_feats.loc[base_row["idx2"], "cell_id"],
        )
        base_lookup[key] = base_row

    rows: list[dict] = []
    for _, pair in combined_pairs.iterrows():
        cid1 = pair["cell_id_r1"]
        cid2 = pair["cell_id_r2"]
        idx1 = id_to_idx1.get(cid1)
        idx2 = id_to_idx2.get(cid2)
        if idx1 is None or idx2 is None:
            continue

        row = {"idx1": idx1, "idx2": idx2, "cell_id_1": cid1, "cell_id_2": cid2}
        base_row = base_lookup.get((cid1, cid2))
        if base_row is not None:
            for extra in ("distance",):
                if extra in base_row and not pd.isna(base_row[extra]):
                    row[extra] = base_row[extra]
        for extra in ("L_pos", "L_nei", "dist_norm", "within_threshold", "patch_x", "patch_y"):
            if extra in pair and not pd.isna(pair[extra]):
                row[extra] = pair[extra]
        rows.append(row)

    print(
        "Topology filtering kept "
        f"{len(trusted_pairs)} trusted anchors and {len(neighbor_filtered)} propagated neighbors "
        f"before 1-to-1 pruning; final matches={len(rows)}."
    )
    return pd.DataFrame(rows).reset_index(drop=True)


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


def estimate_rigid_transform_from_matches_ransac(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
    max_trials: int = 1000,
    residual_threshold: float = 2.0,
    min_inliers: int = 3,
) -> tuple[RigidTransform, np.ndarray]:
    """
    Estimate a robust rigid transform using RANSAC on matched pairs.
    """
    n_matches = len(matches)
    if n_matches == 0:
        raise ValueError("No matches provided to estimate transform.")
    if n_matches < min_inliers:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
        inlier_mask = np.ones(n_matches, dtype=bool)
        return transform, inlier_mask
    pts_fixed = feats1.iloc[matches["idx1"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving = feats2.iloc[matches["idx2"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    robust = estimate_robust_transform(
        pts_fixed,
        pts_moving,
        allow_scale=False,
        prefer_affine=False,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        min_inliers=min_inliers,
    )
    params = np.asarray(robust.transform.params, dtype=float)
    transform = RigidTransform(rotation=params[:2, :2], translation=params[:2, 2])
    return transform, np.asarray(robust.inliers, dtype=bool)


def run_pipeline(
    img1_path: Path = DEFAULT_IMG1_PATH,
    img2_path: Path = DEFAULT_IMG2_PATH,
    top_k: int = DEFAULT_TOP_K,
    feature_weight: float = DEFAULT_FEATURE_WEIGHT,
    topology_weight: float = DEFAULT_TOPOLOGY_WEIGHT,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD,
    spatial_window_size: float | None = DEFAULT_SPATIAL_WINDOW_SIZE,
    napari_view: bool = False,
    segmentation_only: bool = False,
    save_match_table: Path = DEFAULT_SAVE_MATCH_TABLE,
    save_match_overlay_path: Path = DEFAULT_SAVE_MATCH_OVERLAY,
    save_segmentation_prefix: Path = DEFAULT_SAVE_SEGMENTATION_PREFIX,
    save_match_plot_path: Path = DEFAULT_SAVE_MATCH_PLOT,
    save_registration_overlay_path: Path = DEFAULT_SAVE_REGISTRATION_OVERLAY,
    save_registered_moving_path: Path | None = DEFAULT_SAVE_REGISTERED_MOVING,
    save_features_dir: Path | None = DEFAULT_SAVE_FEATURES_DIR,
    residual_prune_quantile: float | None = DEFAULT_RESIDUAL_PRUNE_QUANTILE,
    channel_axis: ChannelAxis = "auto",
    registration_channel: int = -1,
    cellpose_gpu: bool = False,
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
    initial_coarse_transform_path: Path | None = None,
) -> None:
    segmenter = CellposeSegmenter(replace(DEFAULT_CELLPOSE_CONFIG, gpu=cellpose_gpu))

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if channel_axis == "auto":
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
            overlay_img1 = _extract_registration_channel(img1, channel_axis, registration_channel)
            overlay_img2 = _extract_registration_channel(img2, channel_axis, registration_channel)
            masks1, _, _ = segmenter.segment_array(overlay_img1)
            masks2, _, _ = segmenter.segment_array(overlay_img2)
    else:
        overlay_img1 = _extract_registration_channel(img1, channel_axis, registration_channel)
        overlay_img2 = _extract_registration_channel(img2, channel_axis, registration_channel)
        masks1, _, _ = segmenter.segment_array(overlay_img1)
        masks2, _, _ = segmenter.segment_array(overlay_img2)

    if segmentation_only:
        if save_segmentation_prefix is not None:
            from .visualization import save_segmentation_plot

            out1 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img1.tif")
            out2 = save_segmentation_prefix.with_name(save_segmentation_prefix.stem + "_img2.tif")
            save_segmentation_plot(overlay_img1, masks1, out1, title="Segmentation image1")
            save_segmentation_plot(overlay_img2, masks2, out2, title="Segmentation image2")
            print(f"Saved segmentation plots to {out1} and {out2}")

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
    initial_coarse_transform = _load_affine_transform_json(initial_coarse_transform_path)
    if initial_coarse_transform is not None:
        print(f"Loaded initial coarse transform from {initial_coarse_transform_path}")
    two_stage = two_stage_match_cells(
        feats1,
        feats2,
        masks1.shape,
        feature_weight=feature_weight,
        topology_weight=topology_weight,
        position_weight=position_weight,
        top_k=top_k,
        distance_threshold=distance_threshold,
        spatial_window_size=spatial_window_size,
        coarse_top_k=max(24, top_k),
        coarse_distance_threshold=2.0,
        coarse_matching_mode="morphology_guided",
        coarse_allow_scale=False,
        coarse_prefer_affine=False,
        coarse_residual_threshold=max(5.0, float(ransac_residual_threshold) * 2.0),
        coarse_max_trials=min(max(ransac_max_trials, 200), 2000),
        initial_coarse_transform=initial_coarse_transform,
        initial_coarse_transform_method="json_coarse_affine",
    )
    feats2_aligned = two_stage.aligned_df2

    if len(two_stage.coarse_matches) >= 3:
        tx, ty = two_stage.coarse_offset_xy
        if two_stage.coarse_transform_accepted:
            print(
                "Coarse alignment accepted: "
                f"method={two_stage.coarse_transform_method}, "
                f"inliers={two_stage.coarse_inlier_count}/{len(two_stage.coarse_matches)} "
                f"({two_stage.coarse_inlier_ratio:.2f}), "
                f"median_inlier_residual={two_stage.coarse_median_inlier_residual:.2f}px, "
                f"translation=({tx:.2f}, {ty:.2f}) px"
            )
        else:
            print(
                "Coarse alignment rejected: "
                f"method={two_stage.coarse_transform_method}, "
                f"inliers={two_stage.coarse_inlier_count}/{len(two_stage.coarse_matches)} "
                f"({two_stage.coarse_inlier_ratio:.2f}), "
                f"median_inlier_residual={two_stage.coarse_median_inlier_residual:.2f}px. "
                "Fine matching kept original coordinates."
            )
    else:
        print("Coarse alignment skipped; insufficient confident coarse matches.")

    feats1_work = feats1.copy()
    feats2_work = feats2_aligned.copy()

    if use_spatial_clusters:
        feats1_work, feats2_work = assign_clusters_from_round1(
            feats1_work,
            feats2_work,
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

    matches = two_stage.matches.copy()

    if use_spatial_clusters:
        match_cfg = MatchingConfig(
            feature_weight=feature_weight,
            topology_weight=topology_weight,
            position_weight=position_weight,
            top_k=top_k,
            distance_threshold=distance_threshold,
            spatial_window_size=spatial_window_size,
        )
        matches = match_cells_per_cluster(
            feats1_work,
            feats2_work,
            match_cfg,
            cluster_col="cluster_id",
        )

    if use_topology_filtering:
        # Topology filtering requires patch assignments internally
        feats1_topo = assign_patches(feats1_work, w1, h1, x_col="centroid_x", y_col="centroid_y")
        feats2_topo = assign_patches(feats2_work, w1, h1, x_col="centroid_x", y_col="centroid_y")
        if "cell_id" not in feats1_topo.columns:
            feats1_topo["cell_id"] = feats1_topo.index
        if "cell_id" not in feats2_topo.columns:
            feats2_topo["cell_id"] = feats2_topo.index
        matches = rebuild_matches_from_topology_pairs(
            matches,
            feats1_topo,
            feats2_topo,
            image_width=w1,
            image_height=h1,
            k_pos_nei=k_pos_nei,
            k_neighbor=k_neighbor,
            tau_pos=tau_pos,
            tau_nei=tau_nei,
            tau_map=tau_map,
        )

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
        matches = matches.copy()
        matches["ransac_inlier"] = inlier_mask
        inlier_count = int(np.count_nonzero(inlier_mask))
        all_residuals = compute_match_residuals(feats1, feats2, matches, transform)
        if inlier_count >= MIN_MATCHES_FOR_REFINEMENT:
            inlier_residuals = all_residuals[inlier_mask]
            inlier_median = float(np.median(inlier_residuals))
            inlier_mad = float(np.median(np.abs(inlier_residuals - inlier_median)))
            robust_scale = max(1.4826 * inlier_mad, 0.5)
            model_threshold = max(
                float(ransac_residual_threshold) * 2.0,
                inlier_median + 3.0 * robust_scale,
            )
            keep_mask = all_residuals <= model_threshold
            keep_count = int(np.count_nonzero(keep_mask))
            if MIN_MATCHES_FOR_REFINEMENT <= keep_count < len(matches):
                matches = matches.loc[keep_mask].reset_index(drop=True)
                transform, _ = estimate_rigid_transform_from_matches_ransac(
                    feats1,
                    feats2,
                    matches,
                    max_trials=ransac_max_trials,
                    residual_threshold=ransac_residual_threshold,
                    min_inliers=MIN_MATCHES_FOR_REFINEMENT,
                )
                print(
                    "Filtered matches by RANSAC model consistency: "
                    f"kept {keep_count} / {len(keep_mask)} at <= {model_threshold:.2f}px."
                )
        else:
            print(
                "RANSAC found too few inliers to filter matches safely; "
                "keeping the pre-filtered match set."
            )
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
    else:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
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
            if use_ransac_transform:
                transform, _ = estimate_rigid_transform_from_matches_ransac(
                    feats1,
                    feats2,
                    matches,
                    max_trials=ransac_max_trials,
                    residual_threshold=ransac_residual_threshold,
                    min_inliers=MIN_MATCHES_FOR_REFINEMENT,
                )
            else:
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
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

    feats2_registered = feats2.copy()
    coords2_registered = _apply_rigid_to_points(
        feats2_registered[["centroid_x", "centroid_y"]].to_numpy(dtype=float),
        transform.rotation,
        transform.translation,
    )
    feats2_registered["centroid_x"] = coords2_registered[:, 0]
    feats2_registered["centroid_y"] = coords2_registered[:, 1]

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
        from .visualization import save_match_plot, save_aligned_match_plot

        match_plot_path = save_match_plot_path.with_suffix(".tif")
        save_match_plot(overlay_img1, overlay_img2, feats1, feats2, matches, match_plot_path)
        print(f"Saved match plot to {match_plot_path}")
        aligned_match_plot_path = save_match_plot_path.with_name(save_match_plot_path.stem + "_aligned").with_suffix(".tif")
        save_aligned_match_plot(overlay_img1, overlay_img2, feats1, feats2_registered, matches, aligned_match_plot_path)
        print(f"Saved aligned match plot to {aligned_match_plot_path}")

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

    if save_registered_moving_path is not None:
        moving_original = np.asarray(iio.imread(img2_path))
        registered_moving = warp_moving_image_to_fixed(
            moving_original,
            transform,
            output_shape=masks1.shape,
            channel_axis=channel_axis,
            order=1,
        )
        save_registered_moving_path.parent.mkdir(parents=True, exist_ok=True)
        axes = _axes_metadata_for_registered_image(registered_moving, channel_axis)
        metadata = {"axes": axes} if axes is not None else None
        tiff.imwrite(save_registered_moving_path, registered_moving, metadata=metadata)
        print(f"Saved registered moving image to {save_registered_moving_path}")

    if napari_view:
        try:
            import napari  # type: ignore
            from .visualization import launch_napari_viewer, warp_mask_to_image2
        except ImportError:
            print("napari not installed; skipping interactive viewer.")
        else:
            if matches.empty:
                print("No matches to display in napari.")
            else:
                viewer1 = launch_napari_viewer(overlay_img1, masks1, feats1, title="Image1")
                viewer2 = launch_napari_viewer(overlay_img2, masks2, feats2, title="Image2")
                warped = warp_mask_to_image2(
                    masks1, overlay_img2.shape[:2], transform.rotation, transform.translation
                )
                viewer2.add_labels((warped > 0).astype(int), name="warped_mask1_on_image2", opacity=0.4)
                viewer1.window._qt_window.raise_()
                viewer2.window._qt_window.raise_()
                napari.run()
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoAlign legacy compatibility CLI using Cellpose-SAM.")
    parser.add_argument(
        "image1",
        type=Path,
        nargs="?",
        default=DEFAULT_IMG1_PATH,
        help="Path to first image (target).",
    )
    parser.add_argument(
        "image2",
        type=Path,
        nargs="?",
        default=DEFAULT_IMG2_PATH,
        help="Path to second image (source).",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of matches to use.")
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=DEFAULT_FEATURE_WEIGHT,
        help="Weight for morphology features during matching.",
    )
    parser.add_argument(
        "--topology-weight",
        type=float,
        default=DEFAULT_TOPOLOGY_WEIGHT,
        help="Weight for local topology descriptors during matching.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=DEFAULT_POSITION_WEIGHT,
        help="Weight for spatial proximity in matching (0 to disable position).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help="Reject matches whose combined score exceeds this threshold.",
    )
    parser.add_argument(
        "--spatial-window-size",
        type=float,
        default=DEFAULT_SPATIAL_WINDOW_SIZE,
        help="Maximum centroid distance in pixels for a feasible fine-stage match.",
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
    parser.add_argument(
        "--residual-prune-quantile",
        type=float,
        default=None,
        help="Optional quantile in (0,1) for post-fit residual pruning.",
    )
    parser.add_argument(
        "--initial-coarse-transform",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a 3x3 moving->fixed coarse affine. "
            "The original morphology/topology landmark pipeline then runs in the coarse-aligned space."
        ),
    )
    parser.add_argument("--napari", action="store_true", help="Open napari viewers for segmentation results.")
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Run segmentation (and optional napari view) only; skip feature extraction, matching, registration.",
    )
    parser.add_argument(
        "--channel-axis",
        choices=("auto", "first", "last", "none"),
        default="auto",
        help=(
            "Channel layout for 2D multi-channel images. Use 'first' for C,Y,X "
            "DNA FISH images; default 'auto' preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--registration-channel",
        type=int,
        default=-1,
        help="Channel index used for DAPI/nuclear segmentation and landmark registration.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run Cellpose segmentation on GPU.",
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
        "--save-registered-moving",
        type=Path,
        default=DEFAULT_SAVE_REGISTERED_MOVING,
        help="Path to save the moving image warped into fixed-image coordinates.",
    )
    parser.add_argument(
        "--save-features-dir",
        type=Path,
        default=DEFAULT_SAVE_FEATURES_DIR,
        help="Directory to save per-cell feature tables (round1_cells.csv, round2_cells.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Compatibility adapter: the historical positional CLI now delegates to
    # the structured TopoAlign service.  The old import path remains valid for
    # benchmark and user scripts, while new users should call ``topoalign``.
    from .cli_config import MatchingOptions, OutputOptions, TopoAlignConfig, TransformOptions, SegmentationOptions
    from .service import register, segment_image

    try:
        if args.segmentation_only:
            segmentation_dir = args.save_features_dir or Path("outputs/topoalign-segmentation")
            segment_image(
                args.image1,
                segmentation_dir / "fixed",
                channel_axis=args.channel_axis,
                registration_channel=args.registration_channel,
                gpu=args.gpu,
            )
            segment_image(
                args.image2,
                segmentation_dir / "moving",
                channel_axis=args.channel_axis,
                registration_channel=args.registration_channel,
                gpu=args.gpu,
            )
            return 0

        output_dir = args.save_features_dir or Path("outputs/topoalign-run")
        config = TopoAlignConfig(
            fixed=str(args.image1),
            moving=str(args.image2),
            segmentation=SegmentationOptions(
                channel_axis=args.channel_axis,
                registration_channel=args.registration_channel,
                gpu=args.gpu,
            ),
            matching=MatchingOptions(
                top_k=args.top_k,
                feature_weight=args.feature_weight,
                topology_weight=args.topology_weight,
                position_weight=args.position_weight,
                distance_threshold=args.distance_threshold,
                spatial_window_size=args.spatial_window_size,
                use_spatial_clusters=args.use_spatial_clusters,
                n_clusters=args.n_clusters,
                use_topology_filtering=args.use_topology_filtering,
                k_pos_nei=args.k_pos_nei,
                k_neighbor=args.k_neighbor,
                tau_pos=args.tau_pos,
                tau_nei=args.tau_nei,
                tau_map=args.tau_map,
            ),
            transform=TransformOptions(
                method="rigid",
                use_ransac=args.use_ransac_transform,
                ransac_max_trials=args.ransac_max_trials,
                ransac_residual_threshold=args.ransac_residual_threshold,
                initial_coarse_transform=str(args.initial_coarse_transform) if args.initial_coarse_transform else None,
                residual_prune_quantile=args.residual_prune_quantile,
            ),
            output=OutputOptions(output_dir=str(output_dir)),
        )
        register(config)
        return 0
    except (ValueError, FileNotFoundError, ImportError, OSError) as exc:
        print(f"TopoAlign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
