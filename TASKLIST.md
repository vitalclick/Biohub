# First Submission Tasklist
## Biohub – Cell Tracking During Development

**Goal:** Get a valid, scored submission on the Kaggle leaderboard.

---

## Phase 1 – Account & Competition Setup
*Do this on your personal computer/browser. No code required.*

- [ ] **1.1** Create a Kaggle account at https://www.kaggle.com if you don't have one
- [ ] **1.2** Go to the competition URL and click **"Join Competition"**
- [ ] **1.3** Read and accept the competition rules (required before you can submit)
- [ ] **1.4** Verify your phone number on Kaggle (required to enable GPU notebooks)
- [ ] **1.5** On your Kaggle profile → Settings → enable **"Phone Verification"** if not already done

---

## Phase 2 – Understand the Data (30 min)
*Browse the Kaggle Data tab. No downloads needed.*

- [ ] **2.1** Go to the Data tab and browse the file tree to confirm structure:
  - `train/<sample_id>.zarr` + `train/<sample_id>.geff` (image + ground truth)
  - `test/<sample_id>.zarr` (image only)
  - `sample_submission.csv` (reference format)
- [ ] **2.2** Download and inspect `sample_submission.csv` (890 B) — confirm the column names match our notebook
- [ ] **2.3** Note how many test datasets there are (determines how long inference will take)
- [ ] **2.4** Click into one `.zarr` file in the file tree to see the chunk layout (`0/c/{t}/0/0/0`)

---

## Phase 3 – Create Your Kaggle Notebook
*All work happens inside Kaggle — no local setup needed.*

- [ ] **3.1** Go to the competition **Code** tab → click **"New Notebook"**
- [ ] **3.2** In notebook settings (right panel):
  - Set **Language** to Python
  - Set **Accelerator** to GPU (T4 x1) — helps with large volume processing
  - Confirm **Internet** is **OFF** (required for submission eligibility)
- [ ] **3.3** Confirm the competition data is attached:
  - Right panel → **"Add data"** → search for this competition → add it
  - The data should appear at `/kaggle/input/biohub-cell-tracking-during-development/`
- [ ] **3.4** Copy the contents of `cell_tracking_solution.ipynb` from this repo into the Kaggle notebook
  - Either paste cell by cell, or use **File → Import Notebook** and upload the `.ipynb` file

---

## Phase 4 – Validate the Notebook Runs (1–2 hours)
*Run inside Kaggle before submitting.*

- [ ] **4.1** Run the first 3 cells (imports + path checks) and confirm:
  - `Test dir exists: True`
  - At least one dataset is found
- [ ] **4.2** Run `inspect_zarr()` on the first test sample and record:
  - Exact shape `(T, Z, Y, X)`
  - dtype (expect `uint16`)
  - Array path (expect `0/`)
- [ ] **4.3** Update the zarr loading code if the path differs from what we assumed:
  - Our code tries keys `['raw', 'data', '0', 'volume', 'images']` — add the correct key if missing
- [ ] **4.4** Run detection on a **single frame** of one dataset and print the number of detected cells
  - Aim for a plausible count (tens to hundreds of cells per frame, not 0 or 10,000)
- [ ] **4.5** If detection returns 0 cells:
  - Try lowering the `threshold` parameter (e.g., `0.005`) in `detect_cells_log()`
  - Or switch to `detection_method='watershed'`
- [ ] **4.6** Run the full pipeline on **one dataset only** and confirm:
  - Nodes > 0
  - Edges > 0
  - No Python errors or crashes
- [ ] **4.7** Run the full pipeline on **all test datasets**
  - Monitor runtime — must finish within 12 hours total
  - If too slow, see Phase 5 (optimisations)
- [ ] **4.8** Confirm `submission.csv` is written and passes sanity checks (last notebook cell)

---

## Phase 5 – Speed & Quality Fixes (if needed)

### If runtime is too slow (> 10 hours)
- [ ] **5.1** Reduce `num_sigma` in LoG detection from 5 to 3
- [ ] **5.2** Downsample volumes by 2× in Z before detection (cells are anisotropic anyway)
- [ ] **5.3** Process datasets in parallel using Python `multiprocessing` (careful with RAM)
- [ ] **5.4** Skip LoG entirely and use watershed-only (faster but potentially less accurate)

### If cell counts look wrong
- [ ] **5.5** Print the `estimated_number_of_nodes` from the `.geff` metadata of training samples to calibrate expected counts
- [ ] **5.6** Tune `min_sigma` / `max_sigma` to match the apparent cell size in the volumes
- [ ] **5.7** Try different `threshold` values (0.005 → 0.05) and compare detected counts vs ground truth count estimate

### If you get import errors
- [ ] **5.8** Add `pip install <package>` calls at the top of the notebook for any missing library
  - Common issue: `zarr` version — competition uses Zarr v3, ensure `zarr>=2.17` or `zarr>=3.0`
  - Check with: `import zarr; print(zarr.__version__)`

---

## Phase 6 – Submit
*The actual Kaggle submission steps.*

- [ ] **6.1** Click **"Save & Run All (Commit)"** in the top-right of the Kaggle notebook
  - This runs the notebook fresh from top to bottom in a clean environment
  - Do NOT click Submit yet — wait for the commit to finish
- [ ] **6.2** Wait for the commit to complete (green checkmark) — can take 1–12 hours depending on data size
- [ ] **6.3** Once complete, click **"Submit"** button that appears under the notebook output
- [ ] **6.4** Kaggle will validate `submission.csv` — if it errors, read the error message carefully:
  - `Missing dataset`: a test dataset has no rows in your CSV
  - `Invalid column`: column name typo
  - `Wrong types`: non-integer values in node/edge coordinate columns
- [ ] **6.5** If validation passes, your submission will be scored and appear on the **Public Leaderboard**
- [ ] **6.6** Note your initial score — this is your baseline to beat

---

## Phase 7 – Record Your Baseline & Plan Next Steps

- [ ] **7.1** Screenshot or note your leaderboard score (Edge Jaccard + Division Jaccard)
- [ ] **7.2** Compare your score to the top of the leaderboard to estimate the gap
- [ ] **7.3** Check the **Discussion** tab for any public baselines or data insights shared by other participants
- [ ] **7.4** Identify which metric is weaker (Edge Jaccard vs Division Jaccard) — that's where to focus next

---

## Quick Reference

| Item | Value |
|---|---|
| Competition URL | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview |
| Submissions per day | 5 |
| Final submissions for judging | 2 |
| Max notebook runtime | 12 hours |
| Internet in notebook | Must be OFF |
| Output file name | `submission.csv` |
| Voxel scale (z, y, x) | 1.625, 0.40625, 0.40625 µm |
| Max matching distance | 7.0 µm |
| Data volume shape | ~(100, 64, 256, 256) uint16 |

---

## Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `Test dir exists: False` | Data not attached to notebook | Add competition data in right panel |
| `0 cells detected` | Threshold too high or wrong array path | Lower threshold or fix zarr key |
| Notebook times out | Too slow per dataset | Reduce sigma steps, skip LoG |
| Submit button greyed out | Notebook not committed, or internet was ON | Re-commit with internet OFF |
| CSV validation error | Missing dataset or wrong column types | Check sanity cell output |
| zarr load error | Zarr v2 vs v3 incompatibility | `pip install "zarr>=3.0"` at top |
