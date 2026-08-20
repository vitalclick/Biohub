# Biohub – Cell Tracking During Development (Kaggle 2026)

**Competition:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

## Competition goal
Detect, track, and link cells across time in 4-D (T×Z×Y×X) fluorescence microscopy data.
Reconstruct cell lineages including division events.

## Repository structure

| File | Purpose |
|---|---|
| `submission_improved.ipynb` | **Primary submission notebook** — improved pipeline (see below). Locally benchmarked at 0.889 vs 0.657 (strong) / 0.224 (starter) on the same synthetic metric harness. |
| `train_calibration.ipynb` | Run on Kaggle with internet ON: scores candidate configs with the **real metric** against real train `.geff` ground truth, prints the best `CFG` to paste into the submission notebook. |
| `local_eval/` | Offline validation harness: faithful metric implementation, synthetic 4D embryo generator, baseline replicas, benchmark + notebook builder. |
| `strong_start_baseline.ipynb` | Public [STRONG START] baseline (LB 0.581). Kept as reference. |
| `cell_tracking_solution.ipynb` | Earlier exploratory pipeline. Kept for reference. |
| `tracking_utils.py` | Helper module for the exploratory pipeline. |
| `code/` | Public notebooks downloaded from the competition (starter LB 0.143, strong-start LB 0.581). |
| `sample_submission.csv` | Official format reference. |

## Evaluation metric (verified against the organizers' published implementation)

```
score = weighted_adjusted_edge_jaccard + 0.1 × division_jaccard        (max ≈ 1.1)
adjusted_edge_jaccard = max(0, J_edge × (1 − 0.1 × (N_pred − N_true)/N_true))
```

Facts that shape strategy (from `royerlab/kaggle-cell-tracking-competition`):

- **Edge Jaccard dominates** (division term is worth at most 0.1).
- Node matching: per-timepoint optimal assignment on physically scaled distance
  (z=1.625, y=x=0.40625 µm/voxel), capped at **7 µm**.
- **Sparse GT**: predictions with no ground-truth contact are *ignored*, not FPs.
  But a wrong link touching a labeled cell costs double (FP + FN).
- **Over-prediction penalty is mild** (coefficient 0.1): moderate over-detection is
  cheap; starving detection to keep node counts low is a losing trade.
- Predicted edges spanning non-consecutive frames are **dropped** (harmless but
  useless) → gap closing must interpolate a node at the missing frame.
- Out-degree is capped at 2; forks on unlabeled cells are ignored (no FP risk);
  forks on labeled non-dividing cells are FPs; ±1 frame tolerance around the true
  split, with daughter evidence allowed via grandchildren.

## Improved pipeline (`submission_improved.ipynb`)

```
Zarr v3 chunks read directly via zarr.json + blosc2 (no zarr-lib dependency)
        │
        ▼
  Detection (per frame)
  ├── Block-average XY ×4, Z full-res  → physically isotropic grid
  ├── DoG band-pass + robust median+k·MAD threshold  (handles uneven background)
  ├── Local maxima, min separation 1 iso-voxel (blob physics sets resolution)
  └── Full-resolution intensity-weighted centroid refinement  (tighter vs 7 µm cap)
        │
        ▼
  Linking
  ├── Global drift per frame-pair (median of mutual nearest neighbours)
  ├── Per-track constant-velocity prediction, blended with drift
  ├── Hungarian on motion-compensated residuals (gate 10 µm, sanity 18 µm)
  └── Gap closing t→t+2 with interpolated node at t+1
        │
        ▼
  Divisions (gated post-processing; precision-first because weight is only 0.1)
  ├── New track adjacent to a continuing track → adopt as 2nd daughter
  ├── Two new tracks at a dead end → mother-vanished division
  └── Gates: sister ≤7 µm, symmetry about mother ≤3 µm, sister persists ≥3 frames,
      lineage cooldown 4 frames, out-degree ≤2 enforced
        │
        ▼
  submission.csv  (node + edge rows, IDs reset per dataset, sanity-checked)
```

## Local validation results (`local_eval/benchmark.py`)

Synthetic 4-D embryos (4 scenarios: dense/sparse/dim/fast+gappy) scored with the
locally implemented competition metric:

| Pipeline | Edge score | Division J | Combined |
|---|---|---|---|
| Official starter (real LB 0.143) | 0.224 | 0.000 | 0.224 |
| [STRONG START] (real LB 0.581) | 0.657 | 0.000 (10 FP) | 0.657 |
| **Improved (ours)** | **0.885** | 0.034 | **0.889** |

Ablations (contribution to combined score): denser peaks +0.02, gap closing +0.011,
gated divisions +0.003 (vs −0.006 for the ungated version), motion model +0.001,
refinement ≈ neutral on symmetric synthetic blobs (expected to matter more on real
irregular nuclei).

Absolute numbers are inflated vs the real leaderboard (synthetic data is cleaner);
the *ordering and deltas* are what the harness is for. `train_calibration.ipynb`
re-tunes every knob against real train ground truth.

## Workflow

1. Upload `submission_improved.ipynb` to Kaggle → commit → submit (baseline score).
2. Run `train_calibration.ipynb` (internet ON) → paste best `CFG` back into the
   submission notebook → resubmit.
3. Iterate: extend `SWEEP` in the calibration notebook, combine winning deltas.

## Reproducing the local benchmark
```bash
pip install numpy scipy scikit-image pandas blosc2
cd local_eval
python benchmark.py            # full 4-scenario table
python build_notebooks.py      # regenerate both Kaggle notebooks
```

## Submission format
```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_0113de3b,node,1,0,32,128,128,-1,-1
3,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2
```
