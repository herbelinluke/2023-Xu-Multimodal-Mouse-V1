"""
Re-score finished Sensorium runs on validation trials that have real responses.

The Sensorium 2023 release withholds responses for live_test_* and final_test_* trials (they are
all zeros on disk), but the cache and sensorium_trial_split use every trial. This rebuilds each
run's split, loads best_val.pth, and reports mean per-neuron correlation on all val frames
(should match summary.json) and on train/oracle-tier val frames only. Inference only.
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = "/home/herbelinluke/Documents/bioV/2023-Xu-Multimodal-Mouse-V1"
sys.path.insert(0, REPO)
os.chdir(REPO)

from mouse_model.cnn_predictor_transfer import PredictorTransfer
from mouse_model.metrics import StreamingNeuronMetrics
from mouse_model.run_logging import load_runs
from mouse_model.sensorium_cache import SensoriumCachedDataset, build_sensorium_cache
from mouse_model.splits import limit_by_groups, split_by_groups

REAL_TIERS = {"train", "oracle"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = []
_ds = {}

for r in load_runs(dataset="sensorium"):
    if not r["summary"]:
        continue
    cfg = r["config"]
    session = cfg["session"]
    if session not in _ds:
        cache = build_sensorium_cache(os.path.join(os.path.expanduser("~/Downloads"), session))
        ds = SensoriumCachedDataset(cache, response_lag=cfg["split_info"].get("response_lag", 2))
        tiers = ds.meta["tiers"]
        tier_of_trial = dict(zip(ds.meta["trials"], tiers))
        real = np.array([tier_of_trial[t] in REAL_TIERS for t in ds.trial_of_sample])
        _ds[session] = (ds, real)
    ds, real = _ds[session]
    groups = ds.trial_of_sample
    tr, va = split_by_groups(groups, cfg["split_info"]["val_frac"], cfg["seed"])
    tr = limit_by_groups(tr, groups[tr], cfg["max_train_samples"], cfg["seed"])
    assert len(tr) == cfg["split_info"]["n_train"] and len(va) == cfg["split_info"]["n_val"], r["dir"]

    model = PredictorTransfer(num_neurons=ds.num_neurons, use_shifter=cfg.get("shifter", False)).to(device)
    model.load_state_dict(torch.load(os.path.join(r["dir"], "best_val.pth"), map_location=device))
    model.eval()

    loader = DataLoader(Subset(ds, va.tolist()), batch_size=512, shuffle=False, num_workers=6)
    m_all = StreamingNeuronMetrics(ds.num_neurons, device=device)
    m_real = StreamingNeuronMetrics(ds.num_neurons, device=device)
    is_real_val = torch.from_numpy(real[va])
    i0 = 0
    with torch.no_grad():
        for image, behav, spikes in loader:
            image = torch.squeeze(image, 1).to(device)
            pred = model(image, behav.to(device))
            spikes = spikes.to(device)
            m_all.update(pred, spikes)
            mask = is_real_val[i0:i0 + len(pred)].to(device)
            if mask.any():
                m_real.update(pred[mask], spikes[mask])
            i0 += len(pred)
    a, b = m_all.compute(), m_real.compute()
    row = {
        "dir": os.path.basename(r["dir"]), "session": session[:18], "mode": cfg["training_mode"],
        "minutes": cfg["train_minutes_requested"], "stored_cor": r["summary"].get("val_cor_at_best"),
        "recomputed_cor_all_val": a["cor"], "cor_real_val_only": b["cor"],
        "frac_train_real": float(real[tr].mean()), "frac_val_real": float(real[va].mean()),
        "train_real_minutes": float(real[tr].sum() / ds.fps / 60), "val_real_minutes": float(real[va].sum() / ds.fps / 60),
    }
    out.append(row)
    print(json.dumps(row), flush=True)

with open(os.path.join(REPO, "research", "sensorium_rescore_real_trials.json"), "w") as f:
    json.dump(out, f, indent=1)
print("ALL DONE")
