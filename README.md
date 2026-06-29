# Biohub – Cell Tracking During Development (Kaggle 2026)

**Competition:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

## Competition goal
Detect, track, and link cells across time in 4-D (T×Z×Y×X) fluorescence microscopy data.  
Reconstruct cell lineages including division events.

## Repository structure

| File | Purpose |
|---|---|
| `cell_tracking_solution.ipynb` | Main Kaggle submission notebook |
| `tracking_utils.py` | Importable Python module (detection, matching, submission helpers) |

## Pipeline overview

```
Zarr volume (T×Z×Y×X)
        │
        ▼
  Cell Detection (per frame)
  ├── LoG blob detection  (default)
  └── Watershed fallback  (dense/noisy volumes)
        │
        ▼
  Frame-to-frame matching
  └── Hungarian algorithm with division support (1→2 daughters)
        │
        ▼
  Track graph (NetworkX DiGraph)
  └── Division nodes = out-degree ≥ 2
        │
        ▼
  submission.csv
  ├── node rows  (node_id, t, z, y, x)
  └── edge rows  (source_id, target_id)
```

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
pip install zarr scipy scikit-image networkx pandas numpy
jupyter notebook cell_tracking_solution.ipynb
```
