"""
Three tracking pipelines operating on in-memory (T,Z,Y,X) arrays:

- run_starter : replica of the official Kaggle starter (score 0.143)
- run_strong  : replica of the public [STRONG START] baseline (score 0.581)
- run_improved: this repo's improved pipeline

Each returns a metric.Graph (nodes: id -> (t, z, y, x) float voxels).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, label as ndi_label, uniform_filter
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from metric import Graph, SCALE

# ======================================================================
# Official starter replica
# ======================================================================

def run_starter(vol4d, downsample=4, percentile=90, max_link=15.0):
    nodes, edges = {}, []
    nid = 1
    prev = {}  # id -> voxel pos
    for t in range(vol4d.shape[0]):
        ds = vol4d[t][::downsample, ::downsample, ::downsample]
        sm = uniform_filter(ds.astype(np.float32), size=3)
        binary = sm > np.percentile(sm, percentile)
        lab, n = ndi_label(binary)
        cur = {}
        for comp in range(1, n + 1):
            coords = np.argwhere(lab == comp)
            c = coords.mean(axis=0) * downsample
            nodes[nid] = (t, c[0], c[1], c[2])
            cur[nid] = c
            nid += 1
        if prev and cur:
            pids, cids = list(prev), list(cur)
            P = np.array([prev[i] for i in pids]) * SCALE
            C = np.array([cur[i] for i in cids]) * SCALE
            d = np.sqrt(((P[:, None] - C[None, :]) ** 2).sum(2))
            ri, ci = linear_sum_assignment(d)
            for r, c_ in zip(ri, ci):
                if d[r, c_] <= max_link:
                    edges.append((pids[r], cids[c_]))
        prev = cur
    return Graph(nodes=nodes, edges=edges)


# ======================================================================
# [STRONG START] replica
# ======================================================================

def _block_mean_xy(vol, f):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def run_strong(vol4d, xy_ds=4, smooth=1.0, min_peak_dist=3, thresh_rel=0.30,
               max_link=12.0, div_parent=12.0, div_sister=7.0,
               detect_divisions=True):
    nodes, edges = {}, []
    nid = 1
    prev_ids, prev_xyz = [], np.empty((0, 3))
    for t in range(vol4d.shape[0]):
        ds = _block_mean_xy(vol4d[t], xy_ds)
        sm = gaussian_filter(ds, sigma=smooth)
        try:
            otsu = threshold_otsu(sm)
        except Exception:
            otsu = np.percentile(sm, 95)
        bg = float(np.median(sm))
        abs_th = max(otsu, bg + thresh_rel * (float(sm.max()) - bg))
        peaks = peak_local_max(sm, min_distance=min_peak_dist,
                               threshold_abs=abs_th, exclude_border=False)
        cents = peaks.astype(np.float64)
        if cents.size:
            cents[:, 1] = cents[:, 1] * xy_ds + (xy_ds - 1) / 2
            cents[:, 2] = cents[:, 2] * xy_ds + (xy_ds - 1) / 2
        else:
            cents = np.empty((0, 3))
        curr_ids = list(range(nid, nid + len(cents)))
        nid += len(cents)
        for k in range(len(cents)):
            nodes[curr_ids[k]] = (t, cents[k, 0], cents[k, 1], cents[k, 2])

        if len(prev_ids) and len(curr_ids):
            P = prev_xyz * SCALE
            C = cents * SCALE
            D = np.sqrt(((P[:, None] - C[None, :]) ** 2).sum(2))
            BIG = 1e6
            cost = np.where(D <= max_link, D, BIG)
            ri, ci = linear_sum_assignment(cost)
            parent_children = {}
            matched = set()
            for r, c in zip(ri, ci):
                if cost[r, c] < BIG:
                    edges.append((prev_ids[r], curr_ids[c]))
                    parent_children.setdefault(r, []).append(c)
                    matched.add(c)
            if detect_divisions:
                for c in range(len(curr_ids)):
                    if c in matched:
                        continue
                    best_p, best_d = None, np.inf
                    for p in range(len(prev_ids)):
                        if D[p, c] > div_parent or len(parent_children.get(p, [])) != 1:
                            continue
                        sis = parent_children[p][0]
                        sd = float(np.sqrt(((C[c] - C[sis]) ** 2).sum()))
                        if sd <= div_sister and D[p, c] < best_d:
                            best_p, best_d = p, D[p, c]
                    if best_p is not None:
                        edges.append((prev_ids[best_p], curr_ids[c]))
                        parent_children[best_p].append(c)
                        matched.add(c)
        prev_ids, prev_xyz = curr_ids, cents
    return Graph(nodes=nodes, edges=edges)


# ======================================================================
# Improved pipeline
# ======================================================================

DEFAULT_CFG = dict(
    xy_ds=4,
    dog_sigma1=1.1,          # iso-grid voxels (~1.8 um)
    dog_sigma2=2.6,
    peak_min_dist=1,         # iso voxels; let blob physics set the resolution
    thresh_k=5.0,            # peaks: dog > med + k*MAD
    refine=True,             # full-res intensity-weighted centroid refinement
    refine_win=(2, 7, 7),    # +- window (z, y, x) full-res voxels
    max_link=10.0,           # gate on motion-compensated residual (um)
    max_link_abs=18.0,       # absolute sanity gate (um)
    vel_blend=0.5,           # fraction of per-track velocity used in prediction
    detect_divisions=True,
    div_parent=10.0,         # um, parent (t-1) to daughter (t)
    div_sister=7.0,          # um, between daughters at t
    div_symmetry=3.0,        # um, |midpoint(daughters) - parent| gate
    div_sister_minlen=3,     # frames the adopted sister track must persist
    div_cooldown=4,          # frames a lineage cannot fork again
    gap_close=True,
    gap_max_dist=9.0,        # um, predicted-vs-start residual for t->t+2 bridge
    rescue_link=0.0,         # um, 2nd-chance direct link end(t)->start(t+1); 0=off
)


def _detect_improved(vol, cfg):
    """DoG + robust threshold + peaks on isotropic grid; full-res refinement.
    Returns (N,3) float voxel coords."""
    ds = _block_mean_xy(vol, cfg['xy_ds'])
    g1 = gaussian_filter(ds, cfg['dog_sigma1'])
    g2 = gaussian_filter(ds, cfg['dog_sigma2'])
    dog = g1 - g2
    med = float(np.median(dog))
    mad = float(np.median(np.abs(dog - med))) + 1e-6
    th = med + cfg['thresh_k'] * 1.4826 * mad
    peaks = peak_local_max(dog, min_distance=cfg['peak_min_dist'],
                           threshold_abs=th, exclude_border=False)
    if peaks.size == 0:
        return np.empty((0, 3))

    f = cfg['xy_ds']
    cents = peaks.astype(np.float64)
    cents[:, 1] = cents[:, 1] * f + (f - 1) / 2
    cents[:, 2] = cents[:, 2] * f + (f - 1) / 2

    if cfg['refine']:
        Z, Y, X = vol.shape
        wz, wy, wx = cfg['refine_win']
        vf = vol.astype(np.float32)
        out = np.empty_like(cents)
        for i, (cz, cy, cx) in enumerate(cents):
            z0, z1 = max(0, int(cz) - wz), min(Z, int(cz) + wz + 1)
            y0, y1 = max(0, int(cy) - wy), min(Y, int(cy) + wy + 1)
            x0, x1 = max(0, int(cx) - wx), min(X, int(cx) + wx + 1)
            w = vf[z0:z1, y0:y1, x0:x1]
            w = w - w.min()
            s = w.sum()
            if s <= 0:
                out[i] = (cz, cy, cx)
                continue
            gz = (w.sum(axis=(1, 2)) @ np.arange(z0, z1)) / s
            gy = (w.sum(axis=(0, 2)) @ np.arange(y0, y1)) / s
            gx = (w.sum(axis=(0, 1)) @ np.arange(x0, x1)) / s
            out[i] = (gz, gy, gx)
        cents = out
    return cents


def run_improved(vol4d, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    T = vol4d.shape[0]

    dets = [_detect_improved(vol4d[t], cfg) for t in range(T)]

    nodes, edges = {}, []
    nid = 1
    ids_per_t = []
    for t in range(T):
        ids = list(range(nid, nid + len(dets[t])))
        nid += len(ids)
        for k, i in enumerate(ids):
            nodes[i] = (t, dets[t][k, 0], dets[t][k, 1], dets[t][k, 2])
        ids_per_t.append(ids)

    velocity = {}      # node_id -> last displacement (um)
    matched_next = {}  # node_id -> matched successor count
    for t in range(1, T):
        pids, cids = ids_per_t[t - 1], ids_per_t[t]
        if not pids or not cids:
            continue
        P = dets[t - 1] * SCALE
        C = dets[t] * SCALE

        # global drift: median displacement of mutual nearest neighbours
        d_raw = np.sqrt(((P[:, None] - C[None, :]) ** 2).sum(2))
        drift = np.zeros(3)
        if len(pids) >= 3 and len(cids) >= 3:
            nn_pc = d_raw.argmin(axis=1)
            nn_cp = d_raw.argmin(axis=0)
            mutual = [(i, nn_pc[i]) for i in range(len(pids))
                      if nn_cp[nn_pc[i]] == i and d_raw[i, nn_pc[i]] <= 15.0]
            if len(mutual) >= 5:
                disp = np.array([C[j] - P[i] for i, j in mutual])
                drift = np.median(disp, axis=0)

        # motion-compensated predicted positions
        pred = np.empty_like(P)
        for r, pid in enumerate(pids):
            v = velocity.get(pid)
            if v is not None:
                pred[r] = P[r] + cfg['vel_blend'] * v + (1 - cfg['vel_blend']) * drift
            else:
                pred[r] = P[r] + drift

        d_res = np.sqrt(((pred[:, None] - C[None, :]) ** 2).sum(2))
        BIG = 1e6
        cost = np.where((d_res <= cfg['max_link'])
                        & (d_raw <= cfg['max_link_abs']), d_res, BIG)
        ri, ci = linear_sum_assignment(cost)

        parent_children = defaultdict(list)
        matched_c = set()
        for r, c in zip(ri, ci):
            if cost[r, c] < BIG:
                edges.append((pids[r], cids[c]))
                parent_children[r].append(c)
                matched_c.add(c)
                velocity[cids[c]] = C[c] - P[r]
                matched_next[pids[r]] = matched_next.get(pids[r], 0) + 1

    # ---------------- rescue links: end(t) -> start(t+1), wider gate -----
    if cfg['rescue_link'] > 0:
        has_out = defaultdict(int)
        has_in = defaultdict(int)
        for s, t_ in edges:
            has_out[s] += 1
            has_in[t_] += 1
        ends_t = defaultdict(list)
        starts_t = defaultdict(list)
        for n, (t_, z, y, x) in nodes.items():
            if has_out[n] == 0 and t_ < T - 1:
                ends_t[t_].append(n)
            if has_in[n] == 0 and t_ > 0:
                starts_t[t_].append(n)
        for t_ in range(T - 1):
            e_ids, s_ids = ends_t.get(t_, []), starts_t.get(t_ + 1, [])
            if not e_ids or not s_ids:
                continue
            E = np.array([nodes[n][1:] for n in e_ids]) * SCALE
            S = np.array([nodes[n][1:] for n in s_ids]) * SCALE
            pred = np.empty_like(E)
            for k, n in enumerate(e_ids):
                v = velocity.get(n)
                pred[k] = E[k] + (v if v is not None else 0)
            d = np.sqrt(((pred[:, None] - S[None, :]) ** 2).sum(2))
            BIG = 1e6
            cost = np.where(d <= cfg['rescue_link'], d, BIG)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < BIG:
                    edges.append((e_ids[r], s_ids[c]))
                    velocity[s_ids[c]] = S[c] - E[r]

    # ---------------- gap closing (t -> t+2, interpolated node) ----------
    if cfg['gap_close']:
        has_out = defaultdict(int)
        has_in = defaultdict(int)
        for s, t_ in edges:
            has_out[s] += 1
            has_in[t_] += 1
        ends = defaultdict(list)    # t -> node ids ending at t
        starts = defaultdict(list)  # t -> node ids starting at t
        for n, (t_, z, y, x) in nodes.items():
            if has_out[n] == 0 and t_ < T - 1:
                ends[t_].append(n)
            if has_in[n] == 0 and t_ > 0:
                starts[t_].append(n)
        new_nodes = {}
        for t_ in range(T - 2):
            e_ids = ends.get(t_, [])
            s_ids = starts.get(t_ + 2, [])
            if not e_ids or not s_ids:
                continue
            E = np.array([nodes[n][1:] for n in e_ids]) * SCALE
            S = np.array([nodes[n][1:] for n in s_ids]) * SCALE
            # predict end position 2 frames ahead using stored velocity
            pred = np.empty_like(E)
            for k, n in enumerate(e_ids):
                v = velocity.get(n)
                pred[k] = E[k] + (2 * v if v is not None else 0)
            d = np.sqrt(((pred[:, None] - S[None, :]) ** 2).sum(2))
            BIG = 1e6
            cost = np.where(d <= cfg['gap_max_dist'], d, BIG)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= BIG:
                    continue
                n_end, n_start = e_ids[r], s_ids[c]
                mid_vox = (np.array(nodes[n_end][1:]) +
                           np.array(nodes[n_start][1:])) / 2
                m_id = nid; nid += 1
                new_nodes[m_id] = (t_ + 1, mid_vox[0], mid_vox[1], mid_vox[2])
                edges.append((n_end, m_id))
                edges.append((m_id, n_start))
        nodes.update(new_nodes)

    # ---------------- division detection (gated post-processing) ---------
    if cfg['detect_divisions']:
        children = defaultdict(list)
        parents = defaultdict(list)
        for s, t_ in edges:
            children[s].append(t_)
            parents[t_].append(s)

        pos_um = {n: np.array(nodes[n][1:]) * SCALE for n in nodes}
        t_of = {n: nodes[n][0] for n in nodes}

        starts_by_t = defaultdict(list)  # new-track starts
        for n in nodes:
            if not parents[n] and t_of[n] > 0:
                starts_by_t[t_of[n]].append(n)

        def track_len(n, need):
            """Length of the forward chain from n (capped at `need`)."""
            length = 1
            while length < need and len(children[n]) == 1:
                n = children[n][0]
                length += 1
            return length

        def forked_recently(n, k):
            """Any fork within k frames upstream (incl. n)?"""
            steps = 0
            while n is not None and steps <= k:
                if len(children[n]) >= 2:
                    return True
                n = parents[n][0] if parents[n] else None
                steps += 1
            return False

        def mother_pred(u):
            v = velocity.get(u)
            return pos_um[u] + (v if v is not None else 0)

        candidates = []  # (score, parent u, [new starts...])
        for n in nodes:
            t_ = t_of[n]
            kids = children[n]
            if len(kids) >= 2 or t_ >= T - 1:
                continue
            s_list = starts_by_t.get(t_ + 1, [])
            if not s_list:
                continue
            mp = mother_pred(n)
            if len(kids) == 1:
                # continuing track: adopt one nearby new start as 2nd daughter
                c1 = kids[0]
                for s in s_list:
                    dp = np.linalg.norm(pos_um[s] - pos_um[n])
                    if dp > cfg['div_parent']:
                        continue
                    sd = np.linalg.norm(pos_um[s] - pos_um[c1])
                    if sd > cfg['div_sister']:
                        continue
                    sym = np.linalg.norm((pos_um[s] + pos_um[c1]) / 2 - mp)
                    if sym > cfg['div_symmetry']:
                        continue
                    if track_len(s, cfg['div_sister_minlen']) < cfg['div_sister_minlen']:
                        continue
                    if forked_recently(n, cfg['div_cooldown']):
                        continue
                    candidates.append((sym + 0.3 * dp, n, [s]))
            else:
                # ended track: adopt two new starts as both daughters
                near = [s for s in s_list
                        if np.linalg.norm(pos_um[s] - pos_um[n]) <= cfg['div_parent']]
                for i in range(len(near)):
                    for j in range(i + 1, len(near)):
                        s1, s2 = near[i], near[j]
                        sd = np.linalg.norm(pos_um[s1] - pos_um[s2])
                        if sd > cfg['div_sister']:
                            continue
                        sym = np.linalg.norm((pos_um[s1] + pos_um[s2]) / 2 - mp)
                        if sym > cfg['div_symmetry']:
                            continue
                        if (track_len(s1, cfg['div_sister_minlen']) < cfg['div_sister_minlen']
                                or track_len(s2, cfg['div_sister_minlen']) < cfg['div_sister_minlen']):
                            continue
                        if forked_recently(n, cfg['div_cooldown']):
                            continue
                        candidates.append((sym, n, [s1, s2]))

        candidates.sort(key=lambda c: c[0])
        used_starts, used_parents = set(), set()
        for _, u, s_nodes in candidates:
            if u in used_parents or any(s in used_starts for s in s_nodes):
                continue
            if len(children[u]) + len(s_nodes) > 2:
                continue
            for s in s_nodes:
                edges.append((u, s))
                children[u].append(s)
                parents[s].append(u)
                used_starts.add(s)
            used_parents.add(u)

    return Graph(nodes=nodes, edges=edges)
