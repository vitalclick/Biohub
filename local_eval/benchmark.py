"""
Benchmark the three pipelines on synthetic datasets with the local metric.

Usage: python benchmark.py [--quick]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from metric import evaluate_sample, summarise
from pipelines import run_starter, run_strong, run_improved
from synth import generate

QUICK = '--quick' in sys.argv

SCENARIOS = [
    dict(seed=1, n_init=110, z_atten_um=140.0, p_divide=0.006,
         drift_sigma_um=1.2, label_fraction=0.30),           # nominal
    dict(seed=2, n_init=170, z_atten_um=100.0, p_divide=0.008,
         drift_sigma_um=1.8, label_fraction=0.25),           # dense + dimmer deep
    dict(seed=3, n_init=70, z_atten_um=200.0, p_divide=0.004,
         drift_sigma_um=0.8, label_fraction=0.40),           # sparse + clean
    dict(seed=4, n_init=140, z_atten_um=80.0, p_divide=0.010,
         drift_sigma_um=2.4, label_fraction=0.30,
         cell_jitter_um=1.3, dropout_p=0.02),                # hard: fast, dim, gappy
]
if QUICK:
    SCENARIOS = SCENARIOS[:2]

PIPELINES = {
    'starter (official)': lambda v: run_starter(v),
    'strong  (public)': lambda v: run_strong(v),
    'improved (ours)': lambda v: run_improved(v),
}


def main():
    datasets = []
    for sc in SCENARIOS:
        t0 = time.time()
        d = generate(T=24 if QUICK else 36, **sc)
        print(f'synth seed={sc["seed"]}: {d["n_true_nodes"]} true nodes, '
              f'{len(d["sparse_gt"].nodes)} labeled, '
              f'{sum(1 for n, ch in _outdeg(d["sparse_gt"]).items() if ch >= 2)} labeled divisions '
              f'({time.time()-t0:.0f}s)')
        datasets.append(d)

    for name, fn in PIPELINES.items():
        results = []
        t0 = time.time()
        for d in datasets:
            g = fn(d['volume'])
            r = evaluate_sample(g, d['sparse_gt'], d['n_true_nodes'])
            results.append(r)
        agg = summarise(results)
        dt = time.time() - t0
        per = ' | '.join(
            f"J{i}={r['adjusted_edge_jaccard']:.3f}(n={r['n_pred_nodes']}/{r['n_true_nodes']})"
            for i, r in enumerate(results))
        print(f"\n{name}  [{dt:.0f}s]")
        print(f"  edge_score={agg['edge_score']:.4f}  "
              f"div_J={agg['division_jaccard']:.3f} "
              f"(TP={agg['div_tp']} FP={agg['div_fp']} FN={agg['div_fn']})  "
              f"COMBINED={agg['combined_score']:.4f}")
        print(f"  per-sample: {per}")


def _outdeg(g):
    from collections import defaultdict
    out = defaultdict(int)
    for s, t in g.edges:
        out[s] += 1
    return out


if __name__ == '__main__':
    main()
