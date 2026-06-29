# Biohub – Cell Tracking During Development (Kaggle 2026)

**Competition:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

## Competition goal
Detect, track, and link cells across time in 4-D (T×Z×Y×X) fluorescence microscopy data.  
Reconstruct cell lineages including division events.

## Repository structure

| File | Purpose |
|---|---|
| `strong_start_baseline.ipynb` | **Primary baseline** — anisotropy-aware detection + explicit division pass. Validated to run in ~50s on 4 test datasets. Start here. |
| `cell_tracking_solution.ipynb` | Earlier exploratory pipeline (LoG/watershed detection, implicit division via Hungarian doubling). Kept for reference. |
| `tracking_utils.py` | Importable Python module (detection, matching, submission helpers) for `cell_tracking_solution.ipynb` |

## Pipeline overview (`strong_start_baseline.ipynb`)

```
Zarr volume (T×Z×Y×X), read via zarr.json + blosc2 (no zarr-lib version risk)
        │
        ▼
  Cell Detection (per frame)
  ├── Block-average XY by 4× (Z kept full resolution → physically isotropic grid)
  ├── Gaussian smoothing + Otsu/relative threshold
  └── Local-maxima peak detection (avoids connected-components under-segmentation)
        │
        ▼
  Frame-to-frame linking
  ├── Hungarian algorithm on physically-scaled (µm) distances, gated by MAX_LINK_DIST
  └── Explicit division pass: unmatched cells re-checked against single-child
      parents within DIV_PARENT_DIST, confirmed via DIV_SISTER_DIST
        │
        ▼
  submission.csv
  ├── node rows  (node_id, t, z, y, x) — IDs reset per dataset
  └── edge rows  (source_id, target_id) — parent may have 2 outgoing edges (division)
```

**Critical detail driving the design:** the evaluation metric matches cells using
physically-scaled distance with a **7 µm cap**, but Z voxels (1.625 µm) are ~4×
coarser than XY (0.40625 µm). Downsampling all axes equally (as in Kaggle's
official starter notebook) pushes Z error close to the matching cutoff, causing
false negatives. This baseline downsamples **only XY**, keeping Z at full
resolution, which keeps detections inside the matching window.

## Evaluation metric
`Score = 0.5 × Edge Jaccard + 0.5 × Division Jaccard`

- **Edge Jaccard**: TP/(TP+FP+FN) on temporal links, with a node-count penalty
- **Division Jaccard**: micro-averaged across all detected mitosis events

Physical scale: z=1.625 µm/voxel, y=x=0.40625 µm/voxel  
Maximum matching radius: 7.0 µm

## Submission format
```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,edge,-1,-1,-1,-1,-1,1,2
```

## Running locally
```bash
# strong_start_baseline.ipynb (recommended)
pip install blosc2 scipy scikit-image pandas numpy
jupyter notebook strong_start_baseline.ipynb

# cell_tracking_solution.ipynb (alternative/reference)
pip install zarr scipy scikit-image networkx pandas numpy
jupyter notebook cell_tracking_solution.ipynb
```
