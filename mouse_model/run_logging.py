"""
One folder per training run:

  runs/<YYYYmmdd-HHMMSS>_<name>/
    config.json          everything needed to identify / filter the run
    metrics.csv          one row per epoch
    summary.json         best epoch, stop reason, timing (written at the end)
    best_val.pth         checkpoint with the lowest validation loss
    per_neuron_best.npz  per-neuron metrics at the best epoch
    per_neuron_last.npz  per-neuron metrics at the last epoch
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "runs")

METRIC_FIELDS = [
    "epoch", "step", "samples_seen", "epoch_seconds", "elapsed_seconds",
    "train_loss", "val_loss", "val_cor", "val_r2", "val_ev", "val_mse", "val_bits_per_spike",
    "n_valid", "n_neurons", "lr",
]


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class RunLogger:
    def __init__(self, name: str, config: dict, runs_dir: str = DEFAULT_RUNS_DIR):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(runs_dir, f"{stamp}_{name}")
        os.makedirs(self.dir, exist_ok=False)
        self.config = {
            "name": name,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "git": _git_info(),
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            **config,
        }
        self.write_config()
        self._metrics_path = os.path.join(self.dir, "metrics.csv")
        with open(self._metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    @property
    def best_val_path(self) -> str:
        return os.path.join(self.dir, "best_val.pth")

    def update_config(self, **kv) -> None:
        self.config.update(kv)
        self.write_config()

    def write_config(self) -> None:
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(_jsonable(self.config), f, indent=2)

    def log_epoch(self, row: dict) -> None:
        with open(self._metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(
                {k: _jsonable(row.get(k)) for k in METRIC_FIELDS})

    def save_per_neuron(self, tag: str, metrics: dict, epoch: int) -> None:
        keys = ["cor_per_neuron", "r2_per_neuron", "ev_per_neuron", "bits_per_spike_per_neuron", "valid_mask"]
        arrays = {k: (metrics[k].astype(np.float32) if metrics[k].dtype != bool else metrics[k])
                  for k in keys if k in metrics}
        np.savez_compressed(os.path.join(self.dir, f"per_neuron_{tag}.npz"), epoch=epoch, **arrays)

    def write_summary(self, summary: dict) -> None:
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(_jsonable(summary), f, indent=2)


def load_metrics(run_dir: str) -> dict[str, np.ndarray]:
    with open(os.path.join(run_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in (rows[0].keys() if rows else METRIC_FIELDS):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def load_runs(runs_dir: str = DEFAULT_RUNS_DIR, **filters) -> list[dict]:
    """
    Return [{'dir', 'config', 'summary', 'metrics'}] sorted by creation time.
    filters match config keys, e.g. load_runs(dataset="mouse", training_mode="from_scratch").
    A filter value may be a callable predicate.
    """
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        ok = True
        for k, want in filters.items():
            have = cfg.get(k)
            ok = want(have) if callable(want) else have == want
            if not ok:
                break
        if not ok:
            continue
        sum_path = os.path.join(d, "summary.json")
        summary = json.load(open(sum_path)) if os.path.isfile(sum_path) else {}
        runs.append({"dir": d, "config": cfg, "summary": summary, "metrics": load_metrics(d)})
    return runs


def run_label(run: dict, keys=("dataset", "training_mode", "max_train_samples", "seed")) -> str:
    cfg = run["config"]
    return " | ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg)


def plot_runs(runs: list[dict], metric: str = "val_cor", x: str = "epoch", ax=None,
              label_keys=("dataset", "training_mode", "max_train_samples", "seed")):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for r in runs:
        m = r["metrics"]
        if metric in m and x in m:
            ax.plot(m[x], m[metric], label=run_label(r, label_keys))
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def summary_table(runs: list[dict], metric: str = "val_cor",
                  keys=("dataset", "training_mode", "max_train_samples", "seed")) -> list[dict]:
    rows = []
    for r in runs:
        m = r["metrics"]
        vals = m.get(metric, np.array([]))
        best = int(np.nanargmax(vals)) if np.isfinite(vals).any() else None
        rows.append({
            **{k: r["config"].get(k) for k in keys},
            "train_minutes": r["config"].get("split_info", {}).get("train_minutes"),
            f"best_{metric}": None if best is None else float(vals[best]),
            "best_epoch": None if best is None else int(m["epoch"][best]),
            "stop_reason": r["summary"].get("stop_reason"),
            "dir": os.path.basename(r["dir"]),
        })
    return rows
