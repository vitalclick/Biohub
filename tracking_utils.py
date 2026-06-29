"""
Reusable utilities for the Biohub Cell Tracking competition.
Can be imported from the notebook or run standalone for local testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import watershed

# Physical voxel scale (z, y, x) in µm
VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
MAX_MATCH_UM = 7.0


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_cells_watershed(
    volume_3d: np.ndarray,
    smooth_sigma: float = 1.5,
    min_distance_vox: int = 5,
) -> np.ndarray:
    """Return (N,3) integer centroid array (z,y,x) for one 3-D frame."""
    from skimage.feature import peak_local_max

    vol = gaussian_filter(volume_3d.astype(np.float32), sigma=smooth_sigma)

    try:
        thr = threshold_otsu(vol)
    except Exception:
        thr = float(vol.mean())

    mask = vol > thr
    dist = ndi.distance_transform_edt(mask)

    coords = peak_local_max(dist, min_distance=min_distance_vox, labels=mask)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.int32)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, c in enumerate(coords, 1):
        markers[tuple(c)] = i

    labels = watershed(-dist, markers, mask=mask)
    props = regionprops(labels)
    if not props:
        return np.empty((0, 3), dtype=np.int32)
    return np.array([p.centroid for p in props], dtype=np.int32)


def detect_cells_log(
    volume_3d: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 6.0,
    num_sigma: int = 5,
    threshold: float = 0.01,
) -> np.ndarray:
    """LoG blob detection. Returns (N,3) (z,y,x) centroids."""
    from skimage.feature import blob_log

    vol = volume_3d.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax <= vmin:
        return np.empty((0, 3), dtype=np.int32)
    vol = (vol - vmin) / (vmax - vmin)

    blobs = blob_log(
        vol,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        exclude_border=True,
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return blobs[:, :3].astype(np.int32)


# ── Matching ───────────────────────────────────────────────────────────────────

def scaled_distance_matrix(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    voxel_scale: np.ndarray = VOXEL_SCALE,
) -> np.ndarray:
    """Physical-space pairwise Euclidean distances (µm)."""
    a = pts_a * voxel_scale
    b = pts_b * voxel_scale
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def match_frames(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    ids_prev: np.ndarray,
    next_id: int,
    max_dist_um: float = MAX_MATCH_UM,
    max_daughters: int = 2,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """
    Hungarian-based 1-to-N (N<=max_daughters) matching between frames.

    Returns (ids_curr, edges, next_id).
    """
    M = len(pts_curr)
    ids_curr = np.zeros(M, dtype=np.int64)
    edges: list[tuple[int, int]] = []

    if len(pts_prev) == 0 or M == 0:
        ids_curr = np.arange(next_id, next_id + M, dtype=np.int64)
        return ids_curr, edges, next_id + M

    dist = scaled_distance_matrix(pts_prev, pts_curr)
    N = len(pts_prev)

    # Duplicate rows to allow division (1 parent → 2 daughters)
    cost_ext = np.full((N * max_daughters, M), fill_value=1e9)
    for d in range(max_daughters):
        cost_ext[d * N:(d + 1) * N, :] = dist

    row_ind, col_ind = linear_sum_assignment(cost_ext)

    assigned_curr: dict[int, list[int]] = {}
    prev_to_curr: dict[int, list[int]] = {}

    for r, c in zip(row_ind, col_ind):
        orig_prev = r % N
        if cost_ext[r, c] <= max_dist_um:
            assigned_curr.setdefault(c, []).append(orig_prev)
            prev_to_curr.setdefault(orig_prev, []).append(c)

    first_daughter: dict[int, int] = {}  # prev_idx -> first assigned curr_idx

    for ci in range(M):
        if ci in assigned_curr:
            prev_list = assigned_curr[ci]
            best_prev = min(prev_list, key=lambda p: dist[p, ci])
            daughters = prev_to_curr.get(best_prev, [ci])

            if best_prev not in first_daughter:
                first_daughter[best_prev] = ci
                ids_curr[ci] = int(ids_prev[best_prev])
            else:
                # Second daughter: new ID (division event)
                ids_curr[ci] = next_id
                next_id += 1

            edges.append((int(ids_prev[best_prev]), int(ids_curr[ci])))
        else:
            ids_curr[ci] = next_id
            next_id += 1

    return ids_curr, edges, next_id


# ── Submission builder ─────────────────────────────────────────────────────────

def build_submission_rows(
    all_nodes: list[tuple[int, int, int, int, int]],
    all_edges: list[tuple[int, int]],
    dataset_id: str,
) -> list[dict]:
    """Convert node/edge lists to submission row dicts."""
    rows = []
    for node_id, t, z, y, x in all_nodes:
        rows.append({
            'dataset': dataset_id,
            'row_type': 'node',
            'node_id': int(node_id),
            't': int(t), 'z': int(z), 'y': int(y), 'x': int(x),
            'source_id': -1, 'target_id': -1,
        })
    for src, tgt in all_edges:
        rows.append({
            'dataset': dataset_id,
            'row_type': 'edge',
            'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1,
            'source_id': int(src), 'target_id': int(tgt),
        })
    return rows


def finalize_submission(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset DataFrames and add the 'id' index column."""
    submission = pd.concat(all_dfs, ignore_index=True)
    submission.insert(0, 'id', range(len(submission)))
    col_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x',
                 'source_id', 'target_id']
    submission = submission[col_order]
    for col in ['node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']:
        submission[col] = submission[col].astype(int)
    return submission
