"""
Preprocessed, memory-mapped Sensorium cache: one row per valid video frame.

Raw trial files (Sensorium 2023 dynamic export):
  videos/{trial}.npy        (H=36, W=64, T)   time is the LAST axis
  responses/{trial}.npy     (N_neurons, T)    one response bin per video frame
  behavior/{trial}.npy      (2, T)            [pupil dilation, running speed]
  pupil_center/{trial}.npy  (2, T)            [x, y]
Trailing time bins are NaN padding (trials are 240-324 frames long).

build_sensorium_cache() reads every trial once and writes:
  frames.npy     (N, 60, 80) float16   resized, normalized to [0, 1]
  responses.npy  (N, n_neurons) float16 divided by per-neuron std, clipped at 0
  behavior.npy   (N, 6) float32         [speed, 0, 0, pupil_x, pupil_y, dilation]
  trial_id.npy, frame_idx.npy (N,) int32
  trial_has_responses.npy (n_trials,) bool, aligned with meta["trials"]
  meta.json

The competition sessions (hash 9b4f6a...) withhold responses for live_test_* and final_test_*
trials: they are stored as all zeros. Those trials are detected from the data (not from the tier
name, because other releases ship real responses in the same tiers) and are marked in
trial_has_responses.npy; SensoriumCachedDataset.sample_has_responses exposes this per sample.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

CACHE_VERSION = 1
HAS_RESPONSES_FILE = "trial_has_responses.npy"


def default_cache_dir(root_dir: str, target_size=(60, 80), norm_mode: str = "trial") -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = os.path.basename(os.path.normpath(os.path.expanduser(root_dir)))
    return os.path.join(repo_root, "cache", "sensorium", f"{name}_{target_size[0]}x{target_size[1]}_{norm_mode}")


def _list_trials(root_dir: str) -> list[int]:
    vid_dir = os.path.join(root_dir, "data", "videos")
    resp_dir = os.path.join(root_dir, "data", "responses")
    trials = []
    for name in os.listdir(vid_dir):
        if name.endswith(".npy") and name[:-4].isdigit():
            t = int(name[:-4])
            if os.path.isfile(os.path.join(resp_dir, f"{t}.npy")):
                trials.append(t)
    return sorted(trials)


def _valid_length(video: np.ndarray) -> int:
    nan_t = np.isnan(video).any(axis=(0, 1))
    return int(np.argmax(nan_t)) if nan_t.any() else int(video.shape[-1])


def _response_std(root_dir: str, n_neurons: int) -> np.ndarray:
    path = os.path.join(root_dir, "meta", "statistics", "responses", "all", "std.npy")
    if not os.path.isfile(path):
        return np.ones(n_neurons)
    s = np.load(path).astype(np.float64)
    if s.ndim > 1:
        s = np.nanmean(s, axis=1)
    return np.where(s > 1e-12, s, 1.0)


def _normalize(frames: np.ndarray, norm_mode: str) -> np.ndarray:
    """frames: (T, H, W)."""
    if norm_mode == "trial":
        lo, hi = frames.min(), frames.max()
        return (frames - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(frames)
    if norm_mode == "frame":
        lo = frames.min(axis=(1, 2), keepdims=True)
        rng = frames.max(axis=(1, 2), keepdims=True) - lo
        return np.where(rng > 1e-12, (frames - lo) / np.maximum(rng, 1e-12), 0.0)
    if norm_mode == "global255":
        return frames / 255.0
    raise ValueError("norm_mode must be 'trial', 'frame' or 'global255'")


def build_sensorium_cache(
    root_dir: str,
    cache_dir: str | None = None,
    *,
    target_size=(60, 80),
    norm_mode: str = "trial",
    overwrite: bool = False,
) -> str:
    """Build the cache if missing; return its directory."""
    root_dir = os.path.expanduser(root_dir)
    cache_dir = cache_dir or default_cache_dir(root_dir, target_size, norm_mode)
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.isfile(meta_path) and not overwrite:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("version") == CACHE_VERSION:
            return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    trials = _list_trials(root_dir)
    lengths = [_valid_length(np.load(os.path.join(root_dir, "data", "videos", f"{t}.npy"), mmap_mode="r"))
               for t in trials]
    n_total = int(sum(lengths))
    n_neurons = int(np.load(os.path.join(root_dir, "data", "responses", f"{trials[0]}.npy"), mmap_mode="r").shape[0])
    resp_std = _response_std(root_dir, n_neurons)

    open_mm = np.lib.format.open_memmap
    frames_mm = open_mm(os.path.join(cache_dir, "frames.npy"), mode="w+", dtype=np.float16,
                        shape=(n_total, *target_size))
    resp_mm = open_mm(os.path.join(cache_dir, "responses.npy"), mode="w+", dtype=np.float16,
                      shape=(n_total, n_neurons))
    behav = np.zeros((n_total, 6), dtype=np.float32)
    trial_id = np.zeros(n_total, dtype=np.int32)
    frame_idx = np.zeros(n_total, dtype=np.int32)
    has_responses = np.zeros(len(trials), dtype=bool)

    t0 = time.time()
    row = 0
    for k, (t, T) in enumerate(zip(trials, lengths)):
        vid = np.load(os.path.join(root_dir, "data", "videos", f"{t}.npy"))[:, :, :T]
        h, w, _ = vid.shape
        vid = zoom(vid, (target_size[0] / h, target_size[1] / w, 1), order=1)
        frames = _normalize(np.moveaxis(vid, -1, 0), norm_mode)
        frames_mm[row:row + T] = np.clip(np.nan_to_num(frames), 0.0, 1.0).astype(np.float16)

        resp = np.load(os.path.join(root_dir, "data", "responses", f"{t}.npy"))[:, :T].T / resp_std
        resp = np.maximum(np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        resp_mm[row:row + T] = resp.astype(np.float16)
        has_responses[k] = bool(np.any(resp > 0))

        b_path = os.path.join(root_dir, "data", "behavior", f"{t}.npy")
        p_path = os.path.join(root_dir, "data", "pupil_center", f"{t}.npy")
        if os.path.isfile(b_path):
            b = np.nan_to_num(np.load(b_path)[:, :T])
            behav[row:row + T, 0] = b[1]
            behav[row:row + T, 5] = b[0]
        if os.path.isfile(p_path):
            p = np.nan_to_num(np.load(p_path)[:, :T])
            behav[row:row + T, 3] = p[0]
            behav[row:row + T, 4] = p[1]

        trial_id[row:row + T] = t
        frame_idx[row:row + T] = np.arange(T)
        row += T
        if (k + 1) % 50 == 0 or k + 1 == len(trials):
            print(f"  cached {k + 1}/{len(trials)} trials ({row} frames, {time.time() - t0:.0f}s)")

    frames_mm.flush()
    resp_mm.flush()
    del frames_mm, resp_mm
    np.save(os.path.join(cache_dir, "behavior.npy"), behav)
    np.save(os.path.join(cache_dir, "trial_id.npy"), trial_id)
    np.save(os.path.join(cache_dir, "frame_idx.npy"), frame_idx)
    np.save(os.path.join(cache_dir, HAS_RESPONSES_FILE), has_responses)

    tiers_path = os.path.join(root_dir, "meta", "trials", "tiers.npy")
    tiers = np.load(tiers_path).tolist() if os.path.isfile(tiers_path) else None
    meta = {
        "version": CACHE_VERSION,
        "root_dir": root_dir,
        "n_trials": len(trials),
        "n_frames": n_total,
        "n_neurons": n_neurons,
        "target_size": list(target_size),
        "norm_mode": norm_mode,
        "fps": 30.0,
        "trials": trials,
        "trial_lengths": lengths,
        "tiers": None if tiers is None else [str(tiers[t]) for t in trials],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Sensorium cache written to {cache_dir} ({n_total} frames, {n_neurons} neurons)")
    return cache_dir


def trial_has_responses(cache_dir: str) -> np.ndarray:
    """(n_trials,) bool aligned with meta["trials"]: False where every response is zero (withheld).

    Caches built before this flag existed are scanned once (about a minute) and the result is saved.
    """
    path = os.path.join(cache_dir, HAS_RESPONSES_FILE)
    if os.path.isfile(path):
        return np.load(path)
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    resp = np.load(os.path.join(cache_dir, "responses.npy"), mmap_mode="r")
    out = np.zeros(len(meta["trials"]), dtype=bool)
    row = 0
    for k, T in enumerate(meta["trial_lengths"]):
        out[k] = bool(np.any(np.asarray(resp[row:row + T]) > 0))
        row += T
    np.save(path, out)
    return out


class SensoriumCachedDataset(Dataset):
    """
    One sample per video frame, read from build_sensorium_cache() output.

    The response target for frame t is the response bin at t + response_lag within the
    same trial (frames without a target are dropped). Returns (image, behav, spikes) with
    shapes (1, 1, H, W), (1, 6), (n_neurons,) like the mouse dataset with seq_len=1.

    All trials are indexed, including withheld ones; sample_has_responses / sample_tier mark them
    so splits can exclude them (sensorium_trial_split does by default).
    """

    def __init__(self, cache_dir: str, response_lag: int = 1):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.response_lag = int(response_lag)
        self.trial_id_all = np.load(os.path.join(cache_dir, "trial_id.npy"))
        self.frame_idx_all = np.load(os.path.join(cache_dir, "frame_idx.npy"))
        self.behavior = np.load(os.path.join(cache_dir, "behavior.npy"))

        lengths = dict(zip(self.meta["trials"], self.meta["trial_lengths"]))
        trial_len = np.array([lengths[t] for t in self.trial_id_all])
        self.rows = np.nonzero(self.frame_idx_all + self.response_lag < trial_len)[0]
        self.trial_of_sample = self.trial_id_all[self.rows]

        has = dict(zip(self.meta["trials"], trial_has_responses(cache_dir).tolist()))
        tiers = self.meta.get("tiers") or ["unknown"] * len(self.meta["trials"])
        tier = dict(zip(self.meta["trials"], tiers))
        self.sample_has_responses = np.array([has[t] for t in self.trial_of_sample], dtype=bool)
        self.sample_tier = np.array([tier[t] for t in self.trial_of_sample])
        self._frames = None
        self._responses = None

    @property
    def num_neurons(self) -> int:
        return int(self.meta["n_neurons"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_frames"] = None
        state["_responses"] = None
        return state

    def _open(self):
        if self._frames is None:
            self._frames = np.load(os.path.join(self.cache_dir, "frames.npy"), mmap_mode="r")
            self._responses = np.load(os.path.join(self.cache_dir, "responses.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        self._open()
        r = int(self.rows[idx])
        image = torch.from_numpy(self._frames[r].astype(np.float32))[None, None]
        behav = torch.from_numpy(self.behavior[r])[None]
        spikes = torch.from_numpy(self._responses[r + self.response_lag].astype(np.float32))
        return image, behav, spikes
