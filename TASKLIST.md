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
- [ ] **3.4** Copy the contents of `strong_start_baseline.ipynb` from this repo into the Kaggle notebook
  - Either paste cell by cell, or use **File → Import Notebook** and upload the `.ipynb` file
  - This is the recommended baseline — it reads zarr chunks directly via `blosc2` (avoiding zarr-library version issues), uses an anisotropy-aware detector, and includes an explicit division-detection pass

---

## Phase 4 – Validate the Notebook Runs (1–2 hours)
*Run inside Kaggle before submitting.*

- [ ] **4.1** Run the **CONFIG** cell and confirm `TEST_DIR` resolves to a real path
  - Output should print something like `TEST_DIR = /kaggle/input/.../test`
- [ ] **4.2** Run `list_datasets()` / `read_meta()` on the first test sample and record:
  - Number of test datasets found
  - Exact shape `(T, Z, Y, X)` and dtype (expect `uint16`)
- [ ] **4.3** Run `detect()` on a single frame and print the number of detected cells
  - Aim for a plausible count (hundreds to low thousands per frame, not 0)
- [ ] **4.4** If detection returns 0 or too few cells, tune in the CONFIG block:
  - Lower `THRESH_REL` (e.g., 0.30 → 0.15) to detect more/dimmer cells
  - Lower `MIN_PEAK_DIST` to allow denser cell packing
- [ ] **4.5** If detection returns too many cells (noise getting picked up):
  - Raise `THRESH_REL` or `SMOOTH_SIGMA`
- [ ] **4.6** Run the full pipeline (`process_dataset()` loop) on **one dataset only** and confirm:
  - Nodes > 0, Edges > 0, some divisions detected
  - No Python errors or crashes
- [ ] **4.7** Run the full pipeline on **all test datasets**
  - The validated reference run took ~50 seconds for 4 datasets — if yours is far slower, something is off (check accelerator settings, profile the `detect()` call)
- [ ] **4.8** Confirm `submission.csv` is written and passes the built-in sanity-check cell (asserts on columns, dataset coverage, non-negative coords, and no dangling edges)

---

## Phase 5 – Speed & Quality Fixes (if needed)

### If runtime is too slow (unlikely — reference run was ~50s for 4 datasets)
- [ ] **5.1** Increase `XY_DS` (e.g., 4 → 6) for more aggressive downsampling
- [ ] **5.2** Process datasets in parallel using Python `multiprocessing` (careful with RAM)

### If cell counts look wrong (over/under-prediction penalty)
- [ ] **5.3** Print the `estimated_number_of_nodes` from the `.geff` metadata of training samples to calibrate expected counts
- [ ] **5.4** Sweep `MIN_PEAK_DIST` and `THRESH_REL` so your detected node count per frame matches that estimate
- [ ] **5.5** Toggle `DETECT_DIVISIONS = False` to A/B test whether the division pass helps or hurts your score

### If you get import errors
- [ ] **5.6** `blosc2` should be preinstalled on Kaggle; if not, add `!pip install -q blosc2` as the first cell
  - This notebook deliberately avoids the `zarr` package entirely, reading `zarr.json` + raw chunks directly — so zarr v2/v3 mismatches are not a concern here

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
| `TEST_DIR` resolves to wrong/missing path | Data not attached, or mount path changed | Add competition data in right panel; check the `CANDIDATES` fallback list in the CONFIG cell |
| `0 cells detected` | `THRESH_REL` too high | Lower `THRESH_REL` in CONFIG (e.g., 0.30 → 0.15) |
| Too many cells detected (noise) | `THRESH_REL` too low or no smoothing | Raise `THRESH_REL` or `SMOOTH_SIGMA` |
| Submit button greyed out | Notebook not committed, or internet was ON | Re-commit with internet OFF |
| CSV validation error | Missing dataset or wrong column types | Check the built-in sanity-check assertions in the last code cell |
| `blosc2` import error | Package missing on the base image | `!pip install -q blosc2` as the very first cell |
