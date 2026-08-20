"""
Synthetic 4D (T,Z,Y,X) fluorescence-microscopy generator with ground truth.

Mimics the competition data: uint16 volumes of shape ~(T, 64, 256, 256),
anisotropic voxels (z=1.625, y=x=0.40625 um), nuclei as bright blobs that
drift coherently, divide, and dim with imaging depth. Returns the full
lineage graph plus a sparsely-labeled subset (like the competition GT).
"""

from __future__ import annotations

import numpy as np

from metric import Graph, SCALE


def generate(
    T=36, Z=64, Y=256, X=256,
    n_init=110,
    seed=0,
    drift_sigma_um=1.2,        # global drift random walk per frame
    flow_amp_um=1.0,           # coherent flow field amplitude per frame
    cell_jitter_um=0.9,        # per-cell random motion
    p_divide=0.006,            # per-cell per-frame division probability
    sister_sep_um=(2.5, 4.5),  # initial daughter separation
    z_atten_um=140.0,          # exponential attenuation length in z
    dropout_p=0.012,           # per-cell per-frame chance of dimming (missed det.)
    label_fraction=0.3,        # fraction of lineages annotated (sparse GT)
    noise_read=90.0,
    background=350.0,
    peak_intensity=2600.0,
    nucleus_sigma_um=2.1,
):
    rng = np.random.default_rng(seed)
    extent_um = np.array([Z, Y, X]) * SCALE

    # --- simulate trajectories -------------------------------------------
    # cell state: pos (um, 3), intensity factor, lineage root id
    next_id = 1

    class Cell:
        __slots__ = ('id', 'pos', 'inten', 'root', 'dropout')
        def __init__(self, id, pos, inten, root):
            self.id = id
            self.pos = pos
            self.inten = inten
            self.root = root
            self.dropout = False

    cells = []
    for _ in range(n_init):
        pos = rng.uniform([4, 6, 6], extent_um - [4, 6, 6])
        c = Cell(next_id, pos, rng.normal(1.0, 0.22), next_id)
        cells.append(c)
        next_id += 1

    # smooth coherent flow field: low-frequency sinusoids
    phase = rng.uniform(0, 2 * np.pi, size=(3, 3))
    freq = rng.uniform(0.5, 1.5, size=(3, 3)) * 2 * np.pi

    def flow(pos_um, t):
        u = pos_um / extent_um  # 0..1
        out = np.zeros(3)
        for a in range(3):
            out[a] = (np.sin(freq[a, 0] * u[0] + phase[a, 0] + 0.13 * t)
                      + np.sin(freq[a, 1] * u[1] + phase[a, 1])
                      + np.sin(freq[a, 2] * u[2] + phase[a, 2]))
        return out / 3.0 * flow_amp_um

    nodes = {}       # node_id -> (t, z, y, x) voxels (full GT)
    edges = []       # full GT edges
    node_of_cell_prev = {}
    frames_cells = []  # per-frame list of (node_id, pos_um, inten_eff)
    roots = set()

    drift = np.zeros(3)
    for t in range(T):
        drift = drift + rng.normal(0, drift_sigma_um, 3) * [0.4, 1.0, 1.0]
        new_cells = []
        frame_list = []
        for c in cells:
            # motion
            c.pos = (c.pos + drift * 0.15 + flow(c.pos, t)
                     + rng.normal(0, cell_jitter_um, 3) * [0.6, 1.0, 1.0])
            # leave volume?
            if np.any(c.pos < -3) or np.any(c.pos > extent_um + 3):
                continue
            # dropout this frame?
            c.dropout = rng.random() < dropout_p

            vox = c.pos / SCALE
            nid = next_id; next_id += 1
            nodes[nid] = (t, vox[0], vox[1], vox[2])
            roots.add(c.root)
            if c.id in node_of_cell_prev:
                edges.append((node_of_cell_prev[c.id], nid))
            node_of_cell_prev[c.id] = nid
            frame_list.append((nid, c.pos.copy(),
                               c.inten * (0.25 if c.dropout else 1.0)))

            # division?
            if t < T - 1 and rng.random() < p_divide:
                sep = rng.uniform(*sister_sep_um)
                direction = rng.normal(0, 1, 3) * [0.4, 1.0, 1.0]
                direction /= np.linalg.norm(direction) + 1e-9
                d1 = Cell(0, c.pos + direction * sep / 2,
                          c.inten * rng.normal(0.9, 0.05), c.root)
                d2 = Cell(0, c.pos - direction * sep / 2,
                          c.inten * rng.normal(0.9, 0.05), c.root)
                # daughters replace mother; ids managed via node linkage
                d1.id = -nid  # sentinel; will re-link below
                d2.id = -nid - 10**9
                # We link daughters to mother's node at next frame emission:
                node_of_cell_prev[d1.id] = nid
                node_of_cell_prev[d2.id] = nid
                new_cells.extend([d1, d2])
            else:
                new_cells.append(c)
        cells = new_cells
        frames_cells.append(frame_list)

    full_gt = Graph(nodes=nodes, edges=edges)

    # --- sparse labeling: keep a fraction of lineage roots ----------------
    roots = sorted(roots)
    rng.shuffle(roots)
    kept = set(roots[:max(1, int(len(roots) * label_fraction))])

    # find root of each node by traversing from initial cells: instead,
    # track roots via connected components on the full graph
    parent_of = {}
    for s, tt in edges:
        parent_of[tt] = s
    def find_root(n):
        while n in parent_of:
            n = parent_of[n]
        return n
    root_cache = {}
    def root_of(n):
        if n not in root_cache:
            root_cache[n] = find_root(n)
        return root_cache[n]

    # roots set above collected cell.root ids which equal initial node ids?
    # initial cells' first node ids are their first emitted node ids; map:
    first_frame_roots = {root_of(nid) for nid in nodes if nodes[nid][0] == 0}
    first_frame_roots = sorted(first_frame_roots)
    rng.shuffle(first_frame_roots)
    kept = set(first_frame_roots[:max(1, int(len(first_frame_roots) * label_fraction))])

    keep_nodes = {n for n in nodes if root_of(n) in kept}
    sparse_gt = Graph(
        nodes={n: nodes[n] for n in keep_nodes},
        edges=[(s, tt) for s, tt in edges if s in keep_nodes and tt in keep_nodes],
    )

    # --- render volumes ---------------------------------------------------
    sig_vox = nucleus_sigma_um / SCALE  # anisotropic sigmas in voxels
    win = np.ceil(sig_vox * 3).astype(int)
    vol4d = np.zeros((T, Z, Y, X), dtype=np.float32)

    zz = np.arange(Z); yy = np.arange(Y); xx = np.arange(X)
    for t in range(T):
        vol = vol4d[t]
        for nid, pos_um, inten in frames_cells[t]:
            vz, vy, vx = pos_um / SCALE
            z0, z1 = max(0, int(vz - win[0])), min(Z, int(vz + win[0]) + 1)
            y0, y1 = max(0, int(vy - win[1])), min(Y, int(vy + win[1]) + 1)
            x0, x1 = max(0, int(vx - win[2])), min(X, int(vx + win[2]) + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            gz = np.exp(-0.5 * ((zz[z0:z1] - vz) / sig_vox[0]) ** 2)
            gy = np.exp(-0.5 * ((yy[y0:y1] - vy) / sig_vox[1]) ** 2)
            gx = np.exp(-0.5 * ((xx[x0:x1] - vx) / sig_vox[2]) ** 2)
            atten = np.exp(-(pos_um[0]) / z_atten_um)
            blob = (peak_intensity * inten * atten
                    * gz[:, None, None] * gy[None, :, None] * gx[None, None, :])
            vol[z0:z1, y0:y1, x0:x1] += blob
        # background gradient + noise
        bg = background * (1.0 + 0.15 * np.sin(yy / Y * 3.1)[None, :, None]
                                 + 0.1 * np.cos(xx / X * 2.3)[None, None, :])
        vol += bg
        vol += rng.normal(0, noise_read, size=vol.shape).astype(np.float32)
        vol += np.sqrt(np.maximum(vol, 0)) * rng.normal(0, 1.2, size=vol.shape).astype(np.float32)
    np.clip(vol4d, 0, 65535, out=vol4d)
    vol4d = vol4d.astype(np.uint16)

    return {
        'volume': vol4d,
        'full_gt': full_gt,          # complete graph (defines n_true_nodes)
        'sparse_gt': sparse_gt,      # what the metric sees
        'n_true_nodes': len(full_gt.nodes),
    }
