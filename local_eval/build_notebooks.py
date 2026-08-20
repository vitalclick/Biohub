"""
Build the two Kaggle notebooks from shared source blocks so the pipeline
code in the submission and calibration notebooks never drifts apart.

Outputs (repo root):
  - submission_improved.ipynb   (submit this; CPU-only, no internet)
  - train_calibration.ipynb     (run with internet ON; tunes CONFIG on train GT)
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================================================================
# Shared source blocks
# ======================================================================

SRC_IO = r'''
# ---------------------------------------------------------------- I/O --
# Zarr v3 stores are read directly from chunk files with blosc2: shape &
# dtype come from 0/zarr.json, timepoint t lives at 0/c/{t}/0/0/0.
# This avoids any zarr-library version mismatch on the Kaggle image.

def list_datasets(d):
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith('.zarr'))

def read_meta(zarr_path):
    with open(os.path.join(zarr_path, '0', 'zarr.json')) as f:
        m = json.load(f)
    return tuple(m['shape']), np.dtype(m['data_type'])

def load_volume(zarr_path, t, shape, dtype):
    p = os.path.join(zarr_path, '0', 'c', str(t), '0', '0', '0')
    with open(p, 'rb') as f:
        raw = blosc2.decompress(f.read())
    return np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
'''

SRC_DETECT = r'''
# ---------------------------------------------------------- detection --
# Anisotropy-aware: block-average XY by 4 (Z untouched) so the working
# grid is ~physically isotropic (4*0.40625 ~= 1.625 um). Detection is a
# band-pass (DoG) + robust MAD threshold + local maxima, then centroids
# are refined at FULL resolution with an intensity-weighted mean, which
# tightens localisation against the metric's 7 um matching cap.

def block_mean_xy(vol, f):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def detect(vol, cfg):
    """Return (N,3) float voxel-space centroids (z, y, x) for one frame."""
    ds = block_mean_xy(vol, cfg['xy_ds'])
    dog = gaussian_filter(ds, cfg['dog_sigma1']) - gaussian_filter(ds, cfg['dog_sigma2'])
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
            out[i] = ((w.sum(axis=(1, 2)) @ np.arange(z0, z1)) / s,
                      (w.sum(axis=(0, 2)) @ np.arange(y0, y1)) / s,
                      (w.sum(axis=(0, 1)) @ np.arange(x0, x1)) / s)
        cents = out
        cents[:, 0] = np.clip(cents[:, 0], 0, Z - 1)
        cents[:, 1] = np.clip(cents[:, 1], 0, Y - 1)
        cents[:, 2] = np.clip(cents[:, 2], 0, X - 1)
    return cents
'''

SRC_TRACK = r'''
# ----------------------------------------------------------- tracking --
# 1) Frame-to-frame Hungarian assignment on MOTION-COMPENSATED physical
#    distances (global drift from mutual nearest neighbours + per-track
#    velocity), gated by MAX residual.
# 2) Optional rescue pass for broken tracks with a wider gate.
# 3) Gap closing t -> t+2 with an interpolated node at t+1 (edges must
#    connect consecutive frames -- the metric drops skip edges).
# 4) Division detection as gated post-processing: a new track adjacent to
#    an existing one (or two new tracks at a dead end) becomes a second
#    daughter only if sisters are close, roughly symmetric about the
#    mother, persistent, and the lineage has not just forked.

def track_dataset(dets, cfg):
    """dets: list over t of (N,3) float voxel coords.
    Returns (nodes: id->(t,z,y,x), edges: [(src,tgt)])."""
    T = len(dets)
    nodes, edges = {}, []
    nid = 1
    ids_per_t = []
    for t in range(T):
        ids = list(range(nid, nid + len(dets[t])))
        nid += len(ids)
        for k, i in enumerate(ids):
            nodes[i] = (t, dets[t][k, 0], dets[t][k, 1], dets[t][k, 2])
        ids_per_t.append(ids)

    velocity = {}
    for t in range(1, T):
        pids, cids = ids_per_t[t - 1], ids_per_t[t]
        if not pids or not cids:
            continue
        P = dets[t - 1] * SCALE
        C = dets[t] * SCALE
        d_raw = np.sqrt(((P[:, None] - C[None, :]) ** 2).sum(2))
        drift = np.zeros(3)
        if len(pids) >= 3 and len(cids) >= 3:
            nn_pc = d_raw.argmin(axis=1)
            nn_cp = d_raw.argmin(axis=0)
            mutual = [(i, nn_pc[i]) for i in range(len(pids))
                      if nn_cp[nn_pc[i]] == i and d_raw[i, nn_pc[i]] <= 15.0]
            if len(mutual) >= 5:
                drift = np.median(np.array([C[j] - P[i] for i, j in mutual]), axis=0)
        pred = np.empty_like(P)
        for r, pid in enumerate(pids):
            v = velocity.get(pid)
            pred[r] = P[r] + (cfg['vel_blend'] * v + (1 - cfg['vel_blend']) * drift
                              if v is not None else drift)
        d_res = np.sqrt(((pred[:, None] - C[None, :]) ** 2).sum(2))
        BIG = 1e6
        cost = np.where((d_res <= cfg['max_link']) & (d_raw <= cfg['max_link_abs']),
                        d_res, BIG)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < BIG:
                edges.append((pids[r], cids[c]))
                velocity[cids[c]] = C[c] - P[r]

    # rescue pass -------------------------------------------------------
    if cfg['rescue_link'] > 0:
        has_out, has_in = defaultdict(int), defaultdict(int)
        for s, t_ in edges:
            has_out[s] += 1; has_in[t_] += 1
        ends_t, starts_t = defaultdict(list), defaultdict(list)
        for n, (t_, z, y, x) in nodes.items():
            if has_out[n] == 0 and t_ < T - 1: ends_t[t_].append(n)
            if has_in[n] == 0 and t_ > 0: starts_t[t_].append(n)
        for t_ in range(T - 1):
            e_ids, s_ids = ends_t.get(t_, []), starts_t.get(t_ + 1, [])
            if not e_ids or not s_ids:
                continue
            E = np.array([nodes[n][1:] for n in e_ids]) * SCALE
            S = np.array([nodes[n][1:] for n in s_ids]) * SCALE
            pr = np.array([E[k] + velocity.get(n, np.zeros(3))
                           for k, n in enumerate(e_ids)])
            d = np.sqrt(((pr[:, None] - S[None, :]) ** 2).sum(2))
            BIG = 1e6
            cost = np.where(d <= cfg['rescue_link'], d, BIG)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < BIG:
                    edges.append((e_ids[r], s_ids[c]))
                    velocity[s_ids[c]] = S[c] - E[r]

    # gap closing -------------------------------------------------------
    if cfg['gap_close']:
        has_out, has_in = defaultdict(int), defaultdict(int)
        for s, t_ in edges:
            has_out[s] += 1; has_in[t_] += 1
        ends_t, starts_t = defaultdict(list), defaultdict(list)
        for n, (t_, z, y, x) in nodes.items():
            if has_out[n] == 0 and t_ < T - 1: ends_t[t_].append(n)
            if has_in[n] == 0 and t_ > 0: starts_t[t_].append(n)
        new_nodes = {}
        for t_ in range(T - 2):
            e_ids, s_ids = ends_t.get(t_, []), starts_t.get(t_ + 2, [])
            if not e_ids or not s_ids:
                continue
            E = np.array([nodes[n][1:] for n in e_ids]) * SCALE
            S = np.array([nodes[n][1:] for n in s_ids]) * SCALE
            pr = np.array([E[k] + 2 * velocity.get(n, np.zeros(3))
                           for k, n in enumerate(e_ids)])
            d = np.sqrt(((pr[:, None] - S[None, :]) ** 2).sum(2))
            BIG = 1e6
            cost = np.where(d <= cfg['gap_max_dist'], d, BIG)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= BIG:
                    continue
                n_e, n_s = e_ids[r], s_ids[c]
                mid = (np.array(nodes[n_e][1:]) + np.array(nodes[n_s][1:])) / 2
                m_id = nid; nid += 1
                new_nodes[m_id] = (t_ + 1, mid[0], mid[1], mid[2])
                edges.append((n_e, m_id))
                edges.append((m_id, n_s))
        nodes.update(new_nodes)

    # divisions ---------------------------------------------------------
    n_div = 0
    if cfg['detect_divisions']:
        children, parents = defaultdict(list), defaultdict(list)
        for s, t_ in edges:
            children[s].append(t_); parents[t_].append(s)
        pos_um = {n: np.array(nodes[n][1:]) * SCALE for n in nodes}
        t_of = {n: nodes[n][0] for n in nodes}
        starts_by_t = defaultdict(list)
        for n in nodes:
            if not parents[n] and t_of[n] > 0:
                starts_by_t[t_of[n]].append(n)

        def track_len(n, need):
            ln = 1
            while ln < need and len(children[n]) == 1:
                n = children[n][0]; ln += 1
            return ln

        def forked_recently(n, k):
            steps = 0
            while n is not None and steps <= k:
                if len(children[n]) >= 2:
                    return True
                n = parents[n][0] if parents[n] else None
                steps += 1
            return False

        candidates = []
        for n in nodes:
            t_ = t_of[n]
            kids = children[n]
            if len(kids) >= 2 or t_ >= T - 1:
                continue
            s_list = starts_by_t.get(t_ + 1, [])
            if not s_list:
                continue
            mp = pos_um[n]
            if len(kids) == 1:
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
                near = [s for s in s_list
                        if np.linalg.norm(pos_um[s] - pos_um[n]) <= cfg['div_parent']]
                for i in range(len(near)):
                    for j in range(i + 1, len(near)):
                        s1, s2 = near[i], near[j]
                        if np.linalg.norm(pos_um[s1] - pos_um[s2]) > cfg['div_sister']:
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
        used_s, used_p = set(), set()
        for _, u, s_nodes in candidates:
            if u in used_p or any(s in used_s for s in s_nodes):
                continue
            if len(children[u]) + len(s_nodes) > 2:
                continue
            for s in s_nodes:
                edges.append((u, s))
                children[u].append(s)
                parents[s].append(u)
                used_s.add(s)
            used_p.add(u)
            n_div += 1

    return nodes, edges, n_div
'''

SRC_METRIC = r'''
# ------------------------------------------------- competition metric --
# Local re-implementation of the official metric:
#   score = weighted_adjusted_edge_jaccard + 0.1 * division_jaccard
# Matching: per-timepoint optimal assignment, scaled distance <= 7 um.
# Sparse GT: predictions without GT contact are ignored, not FPs.
# Node over-prediction penalty: J * (1 - 0.1*(Npred - Ntrue)/Ntrue).

class G:
    def __init__(self, nodes, edges):
        self.nodes, self.edges = nodes, edges
    def children(self):
        ch = defaultdict(list)
        for s, t in self.edges: ch[s].append(t)
        return ch
    def parents(self):
        pa = defaultdict(list)
        for s, t in self.edges: pa[t].append(s)
        return pa

def _preprocess(pred):
    seen, out, edges = set(), defaultdict(int), []
    for s, t in pred.edges:
        if s not in pred.nodes or t not in pred.nodes: continue
        if (s, t) in seen: continue
        if pred.nodes[t][0] != pred.nodes[s][0] + 1: continue
        if out[s] >= 2: continue
        seen.add((s, t)); out[s] += 1; edges.append((s, t))
    return G(pred.nodes, edges)

def _match(pred, gt):
    bp, bg = defaultdict(list), defaultdict(list)
    for n, v in pred.nodes.items(): bp[v[0]].append(n)
    for n, v in gt.nodes.items(): bg[v[0]].append(n)
    p2g, g2p = {}, {}
    for t in bg:
        gids, pids = bg[t], bp.get(t, [])
        if not pids: continue
        gp = np.array([gt.nodes[g][1:] for g in gids]) * SCALE
        pp = np.array([pred.nodes[p][1:] for p in pids]) * SCALE
        d = np.sqrt(((gp[:, None] - pp[None, :]) ** 2).sum(2))
        cost = np.where(d <= 7.0, d, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < 1e6:
                g2p[gids[r]] = pids[c]; p2g[pids[c]] = gids[r]
    return p2g, g2p

def eval_sample(pred, gt, n_true):
    pred = _preprocess(pred)
    p2g, g2p = _match(pred, gt)
    gch, gpa = gt.children(), gt.parents()
    ges = set(gt.edges)
    tps, fp = set(), 0
    for s, t in pred.edges:
        gs, gtn = p2g.get(s), p2g.get(t)
        if gs is not None and gtn is not None and (gs, gtn) in ges:
            tps.add((gs, gtn)); continue
        bad = (gs is not None and gch.get(gs) and gtn not in gch[gs]) or \
              (gtn is not None and gpa.get(gtn) and gs not in gpa[gtn])
        fp += int(bad)
    tp = len(tps); fn = len(ges) - tp
    den = tp + fp + fn
    j = tp / den if den else 0.0
    ratio = (len(pred.nodes) - n_true) / max(n_true, 1)
    j_adj = max(0.0, j * (1.0 - 0.1 * ratio))

    pch, ppa = pred.children(), pred.parents()
    forks = [n for n, ch in pch.items() if len(ch) >= 2]
    divs = [n for n, ch in gch.items() if len(ch) >= 2]
    evaluable = [f for f in forks if p2g.get(f) is not None and gch.get(p2g[f])]

    def anchor(f, d):
        an = {d} | set(gpa.get(d, []))
        if p2g.get(f) in an: return True
        return any(p2g.get(q) in an for q in ppa.get(f, []))

    def covered(f, d):
        br = pch[f][:2]
        if len(br) < 2: return False
        if any(len(ppa.get(b, [])) > 1 for b in br): return False
        bs = [{b} | set(pch.get(b, [])) for b in br]
        lins = [{c} | set(gch.get(c, [])) for c in gch[d][:2]]
        cov = np.zeros((2, 2), bool)
        for i, ln in enumerate(lins):
            mp = {g2p[g] for g in ln if g in g2p}
            for jx, b in enumerate(bs):
                if mp & b: cov[i, jx] = True
        return (cov[0,0] and cov[1,1]) or (cov[0,1] and cov[1,0])

    cand = {d: [f for f in forks if anchor(f, d) and covered(f, d)] for d in divs}
    pair_d, pair_f = {}, {}
    def assign(d, vis):
        for f in cand[d]:
            if f in vis: continue
            vis.add(f)
            if f not in pair_f or assign(pair_f[f], vis):
                pair_f[f] = d; pair_d[d] = f; return True
        return False
    for d in divs: assign(d, set())
    dtp = len(pair_d); dfn = len(divs) - dtp
    dfp = sum(1 for f in evaluable if f not in pair_f)
    return {'edge_tp': tp, 'edge_fp': fp, 'edge_fn': fn, 'edge_jaccard': j,
            'adjusted_edge_jaccard': j_adj, 'edge_weight': den,
            'div_tp': dtp, 'div_fp': dfp, 'div_fn': dfn,
            'n_pred_nodes': len(pred.nodes), 'n_true_nodes': n_true}

def summarise(rs):
    w = sum(r['edge_weight'] for r in rs)
    edge = (sum(r['adjusted_edge_jaccard'] * r['edge_weight'] for r in rs) / w) if w else 0.0
    dtp = sum(r['div_tp'] for r in rs); dfp = sum(r['div_fp'] for r in rs)
    dfn = sum(r['div_fn'] for r in rs)
    dd = dtp + dfp + dfn
    div = dtp / dd if dd else 0.0
    return {'edge_score': edge, 'division_jaccard': div,
            'combined_score': edge + 0.1 * div,
            'div_tp': dtp, 'div_fp': dfp, 'div_fn': dfn}
'''

CONFIG_BLOCK = r'''
# ----------------------------------------------------------- CONFIG ---
# Defaults were selected on a synthetic benchmark harness (local_eval/ in
# the repo); run train_calibration.ipynb on real train data to re-tune.
SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x um/voxel

CFG = dict(
    xy_ds=4,                 # XY block-average factor (Z kept full-res)
    dog_sigma1=1.1,          # DoG band-pass, iso-grid voxels
    dog_sigma2=2.6,
    peak_min_dist=1,         # min peak separation, iso-grid voxels
    thresh_k=5.0,            # peak threshold: median + k*MAD of DoG
    refine=True,             # full-res intensity-weighted centroid refinement
    refine_win=(2, 7, 7),    # +- refinement window (z, y, x) full-res voxels
    max_link=10.0,           # um, gate on motion-compensated residual
    max_link_abs=18.0,       # um, absolute sanity gate
    vel_blend=0.5,           # per-track velocity weight in motion prediction
    detect_divisions=True,
    div_parent=10.0,         # um, parent->daughter gate
    div_sister=7.0,          # um, sister-sister gate
    div_symmetry=3.0,        # um, |daughter midpoint - mother| gate
    div_sister_minlen=3,     # frames the adopted sister track must persist
    div_cooldown=4,          # frames a lineage cannot fork twice
    gap_close=True,
    gap_max_dist=9.0,        # um, t->t+2 bridge gate (interpolated node)
    rescue_link=0.0,         # um, wider 2nd-chance t->t+1 link; 0 = off
)
'''

# ======================================================================
# Submission notebook
# ======================================================================

sub_cells = []

sub_cells.append(('markdown', r'''# Biohub Cell Tracking — Improved Classical Pipeline

**Approach** (validated on a synthetic benchmark with a local implementation of the
official metric — see `local_eval/` in the project repo):

1. **Detection**: anisotropy-aware DoG band-pass + robust MAD threshold + local maxima
   on a physically isotropic grid (XY block-averaged ×4, Z full-res), then
   **full-resolution intensity-weighted centroid refinement** (tighter localisation
   against the 7 µm matching cap).
2. **Linking**: Hungarian assignment on **motion-compensated** distances
   (global drift via mutual nearest neighbours + per-track velocity).
3. **Gap closing**: t→t+2 bridges with an interpolated node at t+1
   (the metric drops non-consecutive edges, so interpolation is required).
4. **Divisions**: gated post-processing (sister proximity, symmetry about the mother,
   sister-track persistence, lineage cooldown). Metric weight is only 0.1, and false
   forks on labeled cells are penalised — precision matters more than recall.

Local benchmark (4 synthetic datasets, same metric): official starter 0.224,
public strong baseline 0.657, **this pipeline 0.889**.'''))

sub_cells.append(('code', r'''import json, os, time
from collections import defaultdict

import blosc2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
''' + CONFIG_BLOCK + r'''
CANDIDATES = [
    '/kaggle/input/competitions/biohub-cell-tracking-during-development/test',
    '/kaggle/input/biohub-cell-tracking-during-development/test',
]
TEST_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if TEST_DIR is None:
    import glob as _glob
    hits = _glob.glob('/kaggle/input/**/test', recursive=True)
    TEST_DIR = next((h for h in hits if _glob.glob(os.path.join(h, '*.zarr'))),
                    hits[0] if hits else CANDIDATES[0])
print('TEST_DIR =', TEST_DIR)'''))

sub_cells.append(('code', SRC_IO))
sub_cells.append(('code', SRC_DETECT))
sub_cells.append(('code', SRC_TRACK))

sub_cells.append(('code', r'''# ------------------------------------------------------------- run ----
def process_dataset(name):
    zp = os.path.join(TEST_DIR, name + '.zarr')
    shape, dtype = read_meta(zp)
    dets = []
    for t in range(shape[0]):
        vol = load_volume(zp, t, shape, dtype)
        dets.append(detect(vol, CFG))
    nodes, edges, n_div = track_dataset(dets, CFG)
    rows = []
    for nid_, (t, z, y, x) in nodes.items():
        rows.append({'dataset': name, 'row_type': 'node', 'node_id': nid_,
                     't': int(t), 'z': int(round(z)), 'y': int(round(y)),
                     'x': int(round(x)), 'source_id': -1, 'target_id': -1})
    for s, t_ in edges:
        rows.append({'dataset': name, 'row_type': 'edge', 'node_id': -1,
                     't': -1, 'z': -1, 'y': -1, 'x': -1,
                     'source_id': s, 'target_id': t_})
    return rows, len(nodes), len(edges), n_div

datasets = list_datasets(TEST_DIR)
print(f'{len(datasets)} test datasets')
all_rows = []
t0 = time.time()
for i, name in enumerate(datasets, 1):
    rows, nn, ne, nd = process_dataset(name)
    all_rows.extend(rows)
    print(f'[{i}/{len(datasets)}] {name}: {nn} nodes, {ne} edges, '
          f'{nd} divisions ({time.time()-t0:.0f}s elapsed)')
print(f'Total: {time.time()-t0:.0f}s')'''))

sub_cells.append(('code', r'''# ------------------------------------------------- write submission ---
submission = pd.DataFrame(all_rows)
submission.index.name = 'id'
submission.to_csv('submission.csv')
print(f'Wrote submission.csv: {len(submission)} rows '
      f'({(submission.row_type=="node").sum()} nodes, '
      f'{(submission.row_type=="edge").sum()} edges)')

# sanity checks --------------------------------------------------------
assert list(submission.columns) == ['dataset', 'row_type', 'node_id', 't',
                                    'z', 'y', 'x', 'source_id', 'target_id']
assert set(datasets) == set(submission.dataset.unique())
nodes_df = submission[submission.row_type == 'node']
assert (nodes_df[['t', 'z', 'y', 'x']] >= 0).all().all()
for ds, grp in submission.groupby('dataset'):
    nset = set(grp.loc[grp.row_type == 'node', 'node_id'])
    e = grp[grp.row_type == 'edge']
    assert e['source_id'].isin(nset).all() and e['target_id'].isin(nset).all(), ds
    # out-degree <= 2 (metric caps at 2; never emit more)
    from collections import Counter as _C
    assert max(_C(e['source_id']).values(), default=0) <= 2, ds
print('All sanity checks passed')
submission.head()'''))

sub_cells.append(('markdown', r'''## Notes

- `CFG` defaults come from a synthetic benchmark; re-tune with
  `train_calibration.ipynb` (runs the **real metric** against real train GT)
  and paste the winning config here before final submissions.
- Highest-leverage next steps: learned detection (Cellpose/StarDist-3D weights
  attached as a Kaggle dataset), t→t+3 gap closing, division-aware smoothing.'''))

# ======================================================================
# Calibration notebook
# ======================================================================

cal_cells = []

cal_cells.append(('markdown', r'''# Train-Set Calibration — Real-Metric Parameter Tuning

Runs the improved pipeline on **training samples with ground truth** (`.geff`) and
scores every candidate configuration with a local implementation of the official
metric (`score = adjusted_edge_jaccard + 0.1 × division_jaccard`).

**Usage**: run with *Internet ON* (this is not a submission notebook). Set
`N_SAMPLES` below (more = slower but more reliable). At the end, copy the printed
best config into `submission_improved.ipynb`'s `CFG`.'''))

cal_cells.append(('code', r'''import sys, subprocess
# zarr>=3 reads the competition's Zarr v3 stores (train .geff ground truth).
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'zarr>=3.0.0'],
               check=False)
import json, os, time
from collections import defaultdict

import blosc2
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

print('zarr', zarr.__version__)
''' + CONFIG_BLOCK + r'''
CANDIDATES = [
    '/kaggle/input/competitions/biohub-cell-tracking-during-development/train',
    '/kaggle/input/biohub-cell-tracking-during-development/train',
]
TRAIN_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if TRAIN_DIR is None:
    import glob as _glob
    hits = _glob.glob('/kaggle/input/**/train', recursive=True)
    TRAIN_DIR = next((h for h in hits if _glob.glob(os.path.join(h, '*.zarr'))),
                     hits[0] if hits else CANDIDATES[0])
print('TRAIN_DIR =', TRAIN_DIR)

N_SAMPLES = 6            # train samples used for calibration
TEST_DIR = TRAIN_DIR     # I/O helpers below read from this directory'''))

cal_cells.append(('code', SRC_IO))

cal_cells.append(('code', r'''# ------------------------------------------------------ geff reading --
def read_geff(geff_path):
    """Return (gt_nodes: id->(t,z,y,x), gt_edges: [(s,t)], n_true_nodes)."""
    root = zarr.open(geff_path, mode='r')

    def find_array(grp, names):
        for name in names:
            try:
                a = grp[name]
                return np.asarray(a)
            except Exception:
                continue
        return None

    ids = find_array(root, ['nodes/ids'])
    tt = find_array(root, ['nodes/props/t/values'])
    zz = find_array(root, ['nodes/props/z/values'])
    yy = find_array(root, ['nodes/props/y/values'])
    xx = find_array(root, ['nodes/props/x/values'])
    eid = find_array(root, ['edges/ids'])
    if any(v is None for v in (ids, tt, zz, yy, xx, eid)):
        # introspect to help debugging on unexpected layouts
        def walk(g, pre=''):
            for k in g.keys():
                try:
                    sub = g[k]
                except Exception:
                    continue
                if hasattr(sub, 'shape'):
                    print(f'  array {pre}{k}: {sub.shape} {sub.dtype}')
                else:
                    print(f'  group {pre}{k}/')
                    walk(sub, pre + k + '/')
        print('geff layout of', geff_path)
        walk(root)
        raise RuntimeError('unexpected .geff layout - see listing above')

    nodes = {int(i): (int(t), float(z), float(y), float(x))
             for i, t, z, y, x in zip(ids, tt, zz, yy, xx)}
    edges = [(int(s), int(t)) for s, t in np.asarray(eid).reshape(-1, 2)]

    # estimated_number_of_nodes lives in the geff metadata attributes
    n_true = None
    def find_est(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'estimated_number_of_nodes':
                    return v
                r = find_est(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = find_est(v)
                if r is not None:
                    return r
        return None
    try:
        with open(os.path.join(geff_path, 'zarr.json')) as f:
            n_true = find_est(json.load(f))
    except Exception:
        pass
    if n_true is None:
        try:
            n_true = find_est(dict(root.attrs))
        except Exception:
            pass
    if n_true is None:
        n_true = len(nodes)  # fall back to labeled count (upper-bounds penalty)
        print(f'  WARNING: estimated_number_of_nodes not found in {geff_path}; '
              f'using labeled count {n_true}')
    return nodes, edges, int(n_true)

train_sets = sorted(n[:-5] for n in os.listdir(TRAIN_DIR) if n.endswith('.zarr'))
print(f'{len(train_sets)} train datasets; using first {N_SAMPLES}')
train_sets = train_sets[:N_SAMPLES]'''))

cal_cells.append(('code', SRC_DETECT))
cal_cells.append(('code', SRC_TRACK))
cal_cells.append(('code', SRC_METRIC))

cal_cells.append(('code', r'''# -------------------------------------------------- sweep definition --
# One-at-a-time variations around CFG. Detection-affecting keys are grouped
# so per-frame detection is computed once per detection signature.
DET_KEYS = ('xy_ds', 'dog_sigma1', 'dog_sigma2', 'peak_min_dist', 'thresh_k',
            'refine', 'refine_win')

SWEEP = [
    {},                                          # tuned defaults
    {'detect_divisions': False},
    {'peak_min_dist': 2},
    {'thresh_k': 4.0},
    {'thresh_k': 6.5},
    {'dog_sigma1': 0.9, 'dog_sigma2': 2.2},
    {'refine': False},
    {'max_link': 9.0},
    {'max_link': 12.0},
    {'vel_blend': 0.0},
    {'vel_blend': 0.7},
    {'gap_close': False},
    {'gap_max_dist': 12.0},
    {'rescue_link': 14.0},
    {'div_symmetry': 4.5},
    {'div_sister_minlen': 2},
    {'div_parent': 13.0, 'div_sister': 9.0},
]

def det_sig(cfg):
    return tuple(cfg[k] for k in DET_KEYS)

# cache: (dataset, det_sig) -> list of per-frame detections
det_cache = {}

def detections_for(name, cfg):
    key = (name, det_sig(cfg))
    if key not in det_cache:
        zp = os.path.join(TRAIN_DIR, name + '.zarr')
        shape, dtype = read_meta(zp)
        det_cache[key] = [detect(load_volume(zp, t, shape, dtype), cfg)
                          for t in range(shape[0])]
    return det_cache[key]

gt_cache = {}
def gt_for(name):
    if name not in gt_cache:
        gn, ge, ntrue = read_geff(os.path.join(TRAIN_DIR, name + '.geff'))
        gt_cache[name] = (G(gn, ge), ntrue)
    return gt_cache[name]'''))

cal_cells.append(('code', r'''# ------------------------------------------------------- run sweep ----
results = []
t0 = time.time()
for si, delta in enumerate(SWEEP):
    cfg = {**CFG, **delta}
    per_sample = []
    for name in train_sets:
        dets = detections_for(name, cfg)
        nodes, edges, _ = track_dataset(dets, cfg)
        gt, ntrue = gt_for(name)
        per_sample.append(eval_sample(G(nodes, edges), gt, ntrue))
    agg = summarise(per_sample)
    label = ', '.join(f'{k}={v}' for k, v in delta.items()) or 'DEFAULT'
    results.append((agg['combined_score'], agg, label, delta))
    print(f"[{si+1}/{len(SWEEP)}] {label:45s} edge={agg['edge_score']:.4f} "
          f"divJ={agg['division_jaccard']:.3f} "
          f"(TP={agg['div_tp']} FP={agg['div_fp']} FN={agg['div_fn']}) "
          f"COMBINED={agg['combined_score']:.4f}  ({time.time()-t0:.0f}s)")

results.sort(key=lambda r: -r[0])
print('\n=== ranking ===')
for sc, agg, label, delta in results[:8]:
    print(f'{sc:.4f}  {label}')

best = results[0]
print('\nBest config delta over defaults:', best[3] or '(defaults already best)')
print('\nPaste into submission notebook CFG:')
print(json.dumps({**CFG, **best[3]}, indent=2, default=str))'''))

cal_cells.append(('markdown', r'''## Next step

Copy the winning config into `submission_improved.ipynb`'s `CFG` block, commit the
notebook, and submit. If several deltas each beat the default independently, try
combining them in a follow-up sweep (edit `SWEEP` above).'''))


# ======================================================================
# Notebook writer
# ======================================================================

def make_nb(cells):
    out = []
    for kind, src in cells:
        lines = src.strip('\n').splitlines(keepends=True)
        cell = {'cell_type': kind, 'metadata': {}, 'source': lines}
        if kind == 'code':
            cell.update({'execution_count': None, 'outputs': []})
        out.append(cell)
    return {
        'cells': out,
        'metadata': {
            'kaggle': {'accelerator': 'none', 'isGpuEnabled': False,
                       'isInternetEnabled': False, 'language': 'python',
                       'sourceType': 'notebook'},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.13'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


if __name__ == '__main__':
    sub = make_nb(sub_cells)
    with open(os.path.join(ROOT, 'submission_improved.ipynb'), 'w') as f:
        json.dump(sub, f, indent=1)
    cal = make_nb(cal_cells)
    cal['metadata']['kaggle']['isInternetEnabled'] = True
    with open(os.path.join(ROOT, 'train_calibration.ipynb'), 'w') as f:
        json.dump(cal, f, indent=1)
    print('wrote submission_improved.ipynb and train_calibration.ipynb')
