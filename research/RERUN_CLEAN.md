# Clean rerun after the withheld-trial fix

On 2026-11-20 we found that the five competition Sensorium sessions store `live_test_*` and `final_test_*` responses as all zeros, and that every Sensorium run so far had trained and validated on those trials. The code is now fixed. This file explains how to rerun the affected experiments and add the pooled multi-readout runs. Results and background are in `research/RESULTS_active_vs_passive.md`.

## What the fix does

- `mouse_model/sensorium_cache.py` marks each trial whose responses are all zero in `trial_has_responses.npy`. The detection uses the data, not the tier name, because the other Sensorium release (for example `dynamic29156`) ships real responses in the same tiers. Existing caches are scanned once, in a few seconds, with no rebuild. `SensoriumCachedDataset` exposes `sample_has_responses` and `sample_tier`.
- `mouse_model/splits.py`: `sensorium_trial_split(..., exclude_withheld=True)` drops those trials before splitting and records the trials kept and excluded per tier in `split_info`. `exclude_withheld=False` reproduces the old split exactly.
- `mouse_model/pooled_data.py`: `build_pooled_sensorium_datasets` uses the same exclusion. Both pooled builders accept `seed_offset_per_session=False`, which gives every session the same split as the single-animal run with that seed.
- `train_cnn_runs.ipynb` adds `response_filter` to the run key: `"real_only"` for Sensorium and `None` for mouse. `skip_existing` therefore skips the 25 clean mouse scratch runs but not the old Sensorium runs. Those lack the key, so their configs never match. The transfer cell picks the backbone with the same key, so the mouse-target transfers rerun on a clean Sensorium backbone.
- `plot_runs.ipynb` hides contaminated runs (old Sensorium runs, and transfers whose backbone came from one) unless `SHOW_CONTAMINATED = True`. It also gains a table comparing pooled and single-animal results.

Check on session 29515 (seed 0): 406 trials are kept (348 `train`, 58 `oracle`) and 305 are excluded (134 `final_test_bonus`, 57 `final_test_main`, 58 `live_test_bonus`, 56 `live_test_main`). Full training data drops from 81.5 to 47.0 minutes and validation from 35.0 to 20.2 minutes, a factor of 1.73. No trial with all-zero responses remains in either split. The other four sessions show the same pattern (1.73-1.75x).

## Step 1: leave the old runs where they are

We recommend keeping the contaminated runs in `runs/`. They no longer match the run key, so `skip_existing` ignores them, and `plot_runs.ipynb` hides them. Their metadata is committed, and `research/rescore_sensorium_real_trials.py` and the RESULTS file refer to them.

If you prefer them out of the way anyway, move them with git so the history stays consistent. Move the mouse transfers that used a contaminated backbone together with them:

```bash
mkdir -p runs_archive/contaminated_zero_trials
venv/bin/python - <<'EOF'
import json, os, subprocess
from mouse_model.run_logging import load_runs
runs = load_runs()
bad = {os.path.basename(r["dir"]) for r in runs
       if r["config"].get("dataset") == "sensorium" and r["config"].get("response_filter") != "real_only"}
bad |= {os.path.basename(r["dir"]) for r in runs if r["config"].get("pretrained_run") in bad}
for name in sorted(bad):
    subprocess.run(["git", "mv", f"runs/{name}", f"runs_archive/contaminated_zero_trials/{name}"], check=True)
print(len(bad), "runs moved")
EOF
```

`.pth` files are git-ignored, so `git mv` leaves `best_val.pth` behind. Move those files with `mv` afterwards if you want them archived too.

## Step 2: relaunch the sweep

From the repo root (with CUDA, not inside a restricted sandbox):

```bash
cd ~/Documents/bioV/2023-Xu-Multimodal-Mouse-V1
nohup venv/bin/python -u research/run_notebook.py train_cnn_runs.ipynb >> runs/sweep_clean.log 2>&1 &
tail -f runs/sweep_clean.log
```

The notebook runs in this order. Finished runs are skipped, so you can rerun the same command after an interruption.

1. Default animals, scratch learning curve. Mouse J553RT is skipped. Sensorium 29515 runs at 1, 2, 5, 10, 20 and 40 minutes plus full (about 47 minutes).
2. Transfers. Sensorium 29515 from the mouse J553RT backbone (frozen and fine-tune) runs again on real trials. Mouse J553RT from the new clean Sensorium 29515 backbone (frozen and fine-tune) is new.
3. Replication. The other mice and J553RT seeds are skipped. The four other Sensorium sessions run at every size.
4. Pooled multi-readout (shared core, one readout per animal, full data, seed 0, `select_by=val_cor`):
   - `mouse_pooled3_multi_readout_full_s0`: all 3 Xu mice, about 72 minutes of training data in total.
   - `sensorium_pooled2_multi_readout_full_s0`: sessions 29515 and 29623, about 92 real minutes.
   - The 5-session Sensorium run (about 236 real minutes) is commented out in the last cell. Uncomment it once the 2-session run looks healthy.

A CPU dry run with stubbed trainers found 41 runs to launch: 35 Sensorium scratch, 4 transfers and 2 pooled. Expect roughly 8-12 hours on the RTX 2060 Super. The Sensorium runs are about 40% shorter than before because they now see only real trials.

**RAM.** Each Sensorium cache is about 5 GB and memory-mapped, so nothing is loaded whole. A single session or the 2-session pooled run fits in the page cache of the 16 GB machine. The 5-session pooled run reads from about 26 GB of caches. It will work, but it becomes disk-bound and slower, and swap filled up during earlier single-session runs. The pooled cell already uses 4 data-loader workers instead of 8. Close other memory-heavy programs before the 5-session run.

## Step 3: replot

```bash
MPLBACKEND=Agg venv/bin/python research/run_notebook.py plot_runs.ipynb
```

This rewrites `research/run_figures/*.png`, prints the learning-curve table and fit, and prints pooled versus single-animal per-session correlations. Then update the numbers in `research/RESULTS_active_vs_passive.md`. All Sensorium and transfer numbers there are currently contaminated.
