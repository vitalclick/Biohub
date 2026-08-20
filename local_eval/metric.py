"""
Local implementation of the Biohub cell-tracking competition metric.

Faithful re-implementation of the documented metric
(https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md):

  score = weighted_adjusted_edge_jaccard + 0.1 * division_jaccard

- Nodes matched per timepoint via optimal bipartite assignment on
  physically scaled centroid distance, capped at 7.0 um.
- Edge TP: predicted edge whose endpoints match GT nodes joined by a GT edge.
- Edge FP: predicted edge with at least one matched endpoint whose GT
  connectivity contradicts the prediction. Predictions in unlabeled
  territory are ignored (sparse GT).
- Edge FN: GT edge not recovered.
- Adjusted J = max(0, J * (1 - 0.1*(T_pred - T_true)/T_true)).
- Divisions: fork (out-degree >= 2) matched to GT divisions within a
  +-1 frame anchor window, requiring two distinct daughter branches.
  Forks on matched non-dividing GT nodes count as FP; forks on
  unmatched nodes or childless GT nodes are ignored.
- Preprocessing mirrors the official code: drop non-consecutive edges,
  drop duplicate edges, cap out-degree at 2.

Division evaluation here is a close approximation of the official
`division_metrics.evaluate_divisions` (grandchild-fallback subtleties are
simplified); edge evaluation follows the official description exactly.
Intended for *relative* comparison of pipelines offline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x um/voxel
MAX_MATCH_UM = 7.0
ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1


@dataclass
class Graph:
    """nodes: id -> (t, z, y, x) in voxels; edges: list[(src, tgt)]."""
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def children(self):
        ch = defaultdict(list)
        for s, t in self.edges:
            ch[s].append(t)
        return ch

    def parents(self):
        pa = defaultdict(list)
        for s, t in self.edges:
            pa[t].append(s)
        return pa


def _preprocess_pred(pred: Graph) -> Graph:
    """Mirror official preprocessing: consecutive-frame edges only,
    dedup, out-degree cap of 2 (keep earliest edges)."""
    seen = set()
    edges = []
    out_count = defaultdict(int)
    for s, t in pred.edges:
        if s not in pred.nodes or t not in pred.nodes:
            continue
        if (s, t) in seen:
            continue
        if pred.nodes[t][0] != pred.nodes[s][0] + 1:
            continue  # non-consecutive: dropped (not FP)
        if out_count[s] >= 2:
            continue  # out-degree cap
        seen.add((s, t))
        out_count[s] += 1
        edges.append((s, t))
    return Graph(nodes=pred.nodes, edges=edges)


def _match_nodes(pred: Graph, gt: Graph):
    """Per-timepoint optimal bipartite matching on scaled distance <= 7 um.
    Returns pred_to_gt, gt_to_pred dicts."""
    by_t_pred = defaultdict(list)
    by_t_gt = defaultdict(list)
    for nid, (t, z, y, x) in pred.nodes.items():
        by_t_pred[t].append(nid)
    for nid, (t, z, y, x) in gt.nodes.items():
        by_t_gt[t].append(nid)

    p2g, g2p = {}, {}
    for t in by_t_gt:
        gids = by_t_gt[t]
        pids = by_t_pred.get(t, [])
        if not pids:
            continue
        gp = np.array([gt.nodes[g][1:] for g in gids], dtype=np.float64) * SCALE
        pp = np.array([pred.nodes[p][1:] for p in pids], dtype=np.float64) * SCALE
        d = np.sqrt(((gp[:, None] - pp[None, :]) ** 2).sum(axis=2))
        big = 1e6
        cost = np.where(d <= MAX_MATCH_UM, d, big)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < big:
                g2p[gids[r]] = pids[c]
                p2g[pids[c]] = gids[r]
    return p2g, g2p


def evaluate_edges(pred: Graph, gt: Graph, p2g: dict, g2p: dict):
    """Return (tp, fp, fn)."""
    gt_children = gt.children()
    gt_parents = gt.parents()
    gt_edge_set = set(gt.edges)

    tp_edges = set()
    fp = 0
    for s, t in pred.edges:
        gs, gt_ = p2g.get(s), p2g.get(t)
        if gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set:
            tp_edges.add((gs, gt_))
            continue
        # FP if either matched endpoint has contradicting GT connectivity
        is_fp = False
        if gs is not None and gt_children.get(gs):
            if gt_ not in gt_children[gs]:
                is_fp = True
        if not is_fp and gt_ is not None and gt_parents.get(gt_):
            if gs not in gt_parents[gt_]:
                is_fp = True
        fp += int(is_fp)
        # else: unlabeled territory -> ignored

    tp = len(tp_edges)
    fn = len(gt_edge_set) - tp
    return tp, fp, fn


def evaluate_divisions(pred: Graph, gt: Graph, p2g: dict, g2p: dict):
    """Approximate official division evaluation. Return (tp, fp, fn)."""
    pred_children = pred.children()
    pred_parents = pred.parents()
    gt_children = gt.children()
    gt_parents = gt.parents()

    pred_forks = [n for n, ch in pred_children.items() if len(ch) >= 2]
    gt_divs = [n for n, ch in gt_children.items() if len(ch) >= 2]

    # Evaluable forks: fork node matched to a GT node with out-degree >= 1.
    # Fork matched to childless GT node, or unmatched -> ignored.
    evaluable = []
    for f in pred_forks:
        g = p2g.get(f)
        if g is not None and gt_children.get(g):
            evaluable.append(f)

    # Candidate forks per GT division (anchor within +-1 frame of split)
    def anchor_ok(f, d):
        """f anchored on d or parent(d), directly or via f's predecessor."""
        anchors = {d}
        for p in gt_parents.get(d, []):
            anchors.add(p)
        if p2g.get(f) in anchors:
            return True
        for fp_ in pred_parents.get(f, []):
            if p2g.get(fp_) in anchors:
                return True
        return False

    def daughters_covered(f, d):
        """Two GT daughter lineages must map into two distinct branches of f."""
        branches = pred_children[f][:2]
        if len(branches) < 2:
            return False
        # merged branches invalid
        for b in branches:
            if len(pred_parents.get(b, [])) > 1:
                return False
        # branch descendant sets (branch + its children)
        bsets = []
        for b in branches:
            s = {b}
            s.update(pred_children.get(b, []))
            bsets.append(s)
        # GT daughter lineages (child + grandchildren)
        lineages = []
        for c in gt_children[d][:2]:
            lin = {c}
            lin.update(gt_children.get(c, []))
            lineages.append(lin)
        # evidence matrix: lineage i covered by branch j?
        cov = np.zeros((2, 2), dtype=bool)
        for i, lin in enumerate(lineages):
            matched_pred = {g2p[g] for g in lin if g in g2p}
            for j, bs in enumerate(bsets):
                if matched_pred & bs:
                    cov[i, j] = True
        # need a perfect matching lineage->distinct branch
        return (cov[0, 0] and cov[1, 1]) or (cov[0, 1] and cov[1, 0])

    cand = defaultdict(list)  # gt_div -> [forks]
    for d in gt_divs:
        for f in pred_forks:
            if anchor_ok(f, d) and daughters_covered(f, d):
                cand[d].append(f)

    # maximum bipartite pairing (greedy augmenting; counts are small)
    pair_d = {}
    pair_f = {}

    def try_assign(d, visited):
        for f in cand[d]:
            if f in visited:
                continue
            visited.add(f)
            if f not in pair_f or try_assign(pair_f[f], visited):
                pair_f[f] = d
                pair_d[d] = f
                return True
        return False

    for d in gt_divs:
        try_assign(d, set())

    tp = len(pair_d)
    fn = len(gt_divs) - tp
    fp = sum(1 for f in evaluable if f not in pair_f)
    return tp, fp, fn


def evaluate_sample(pred: Graph, gt: Graph, n_true_nodes: int):
    """Evaluate one dataset. n_true_nodes = estimate of TOTAL cell count
    (estimated_number_of_nodes), not the sparse labeled count."""
    pred = _preprocess_pred(pred)
    p2g, g2p = _match_nodes(pred, gt)
    etp, efp, efn = evaluate_edges(pred, gt, p2g, g2p)
    dtp, dfp, dfn = evaluate_divisions(pred, gt, p2g, g2p)

    denom = etp + efp + efn
    j = etp / denom if denom else 0.0
    ratio = (len(pred.nodes) - n_true_nodes) / max(n_true_nodes, 1)
    j_adj = max(0.0, j * (1.0 - ADJUSTMENT_ALPHA * ratio))
    return {
        'edge_tp': etp, 'edge_fp': efp, 'edge_fn': efn,
        'edge_jaccard': j, 'adjusted_edge_jaccard': j_adj,
        'edge_weight': denom,
        'div_tp': dtp, 'div_fp': dfp, 'div_fn': dfn,
        'n_pred_nodes': len(pred.nodes), 'n_true_nodes': n_true_nodes,
    }


def summarise(sample_results: list) -> dict:
    """Aggregate: edge J weight-averaged by (TP+FP+FN); divisions micro."""
    wsum = sum(r['edge_weight'] for r in sample_results)
    if wsum:
        edge = sum(r['adjusted_edge_jaccard'] * r['edge_weight']
                   for r in sample_results) / wsum
    else:
        edge = 0.0
    dtp = sum(r['div_tp'] for r in sample_results)
    dfp = sum(r['div_fp'] for r in sample_results)
    dfn = sum(r['div_fn'] for r in sample_results)
    ddenom = dtp + dfp + dfn
    div = dtp / ddenom if ddenom else 0.0
    return {
        'edge_score': edge,
        'division_jaccard': div,
        'combined_score': edge + SCORE_DIVISION_WEIGHT * div,
        'div_tp': dtp, 'div_fp': dfp, 'div_fn': dfn,
    }
