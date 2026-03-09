"""
PyTorch Dataset for Sensorium competition-style data.
Data layout: data/{videos,responses,behavior,pupil_center}/{trial_id}.npy, meta/trials/tiers.npy.
Returns (image, behav, responses) compatible with train_cnn_shifter_table_1 training loop.

Raw file axes:
  responses: (N_neurons, T_bins) – e.g. (7863, 324)
  behavior:  (N_features, T_bins) – e.g. (2, 324)
  videos:    (N_frames, H, W)     – e.g. (36, 64, 324)

vid_frame modes:
  "mean"      – one sample per trial (mean frame → time-averaged response per neuron).
  "middle"    – one sample per trial (middle frame → time-averaged response per neuron).
  "per_frame" – one sample per (trial, frame). Each video frame is paired with the
                neural responses averaged within that frame's time window.
                Analogous to the mouse dataset (one frame → one spike vector).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


def normalize_movie(movie):
    """Normalize the range of gray levels to [0, 1] (NaN-safe)."""
    norm_movie = movie.astype(float)
    vmin = np.nanmin(norm_movie)
    norm_movie -= vmin
    vmax = np.nanmax(norm_movie)
    if not np.isclose(vmax, 0):
        norm_movie /= vmax
    return norm_movie


def resize_video_frame(frame, target_shape=(60, 80)):
    """Resize a single frame to target (H, W)."""
    if frame.shape == target_shape:
        return frame
    h, w = frame.shape
    th, tw = target_shape
    zoom_factors = (th / h, tw / w)
    return zoom(frame, zoom_factors, order=1)


def _bin_responses_to_frames(resp, n_frames):
    """Bin (T_bins, N_neurons) responses into (n_frames, N_neurons) by averaging within each bin.

    Caller must pass responses transposed to (T_bins, N_neurons) if the raw
    file stores them as (N_neurons, T_bins).
    """
    t_resp, n_neurons = resp.shape
    bin_size = t_resp / n_frames
    out = np.empty((n_frames, n_neurons), dtype=np.float64)
    for f in range(n_frames):
        start = int(round(f * bin_size))
        end = int(round((f + 1) * bin_size))
        end = max(end, start + 1)
        out[f] = np.nanmean(resp[start:end], axis=0)
    return out


def _bin_behavior_to_frames(behav_2d, n_frames):
    """Bin (n_features, T_behav) behavior into (n_frames, n_features)."""
    n_features, t_behav = behav_2d.shape
    bin_size = t_behav / n_frames
    out = np.empty((n_frames, n_features), dtype=np.float64)
    for f in range(n_frames):
        start = int(round(f * bin_size))
        end = int(round((f + 1) * bin_size))
        end = max(end, start + 1)
        out[f] = np.nanmean(behav_2d[:, start:end], axis=-1)
    return out


class SensoriumDataset(Dataset):
    """
    Sensorium dataset.

    vid_frame="mean"/"middle": one sample per trial (original behaviour).
    vid_frame="per_frame": one sample per (trial, video frame), like the mouse dataset.
    """

    TIER_TRAIN = 0
    TIER_VAL = 1
    TIER_TEST = 2

    def __init__(
        self,
        root_dir,
        data_split="train",
        seq_len=1,
        vid_frame="mean",
        standardize_responses=True,
        target_size=(60, 80),
    ):
        self.root_dir = os.path.expanduser(root_dir)
        self.seq_len = seq_len
        self.vid_frame = vid_frame
        self.standardize_responses = standardize_responses
        self.target_size = target_size

        data_videos = os.path.join(self.root_dir, "data", "videos")
        data_responses = os.path.join(self.root_dir, "data", "responses")
        self._missing = []

        # ---- resolve trial indices from data_split ----
        self.trial_indices = self._resolve_trial_indices(data_split, data_videos, data_responses)

        # ---- response standardisation ----
        self._num_neurons = None
        self._response_std = None
        if self.standardize_responses and len(self.trial_indices) > 0:
            self._response_std = self._load_response_std()

        if len(self.trial_indices) > 0:
            r0 = np.load(os.path.join(data_responses, f"{self.trial_indices[0]}.npy"))
            self._num_neurons = r0.shape[0] if r0.ndim > 1 else int(r0.size)

        # ---- per_frame: detect n_frames and build flat index ----
        self._n_frames = None
        if self.vid_frame == "per_frame" and len(self.trial_indices) > 0:
            v0 = np.load(os.path.join(data_videos, f"{self.trial_indices[0]}.npy"))
            self._n_frames = v0.shape[0] if v0.ndim == 3 else 1

    # ------------------------------------------------------------------
    # Trial-index resolution (all the tier / split logic lives here)
    # ------------------------------------------------------------------
    def _resolve_trial_indices(self, data_split, data_videos, data_responses):
        ds = str(data_split).lower()

        if ds == "all":
            if not os.path.isdir(data_videos):
                raise FileNotFoundError(f"videos dir not found: {data_videos}")
            valid = []
            for name in os.listdir(data_videos):
                if not name.endswith(".npy"):
                    continue
                trial_id = name[:-4]
                try:
                    trial_id = int(trial_id)
                except ValueError:
                    pass
                if os.path.isfile(os.path.join(data_responses, f"{trial_id}.npy")):
                    valid.append(trial_id)
            return sorted(valid)

        tiers_path = os.path.join(self.root_dir, "meta", "trials", "tiers.npy")
        if not os.path.isfile(tiers_path):
            raise FileNotFoundError(f"tiers.npy not found at {tiers_path}")
        tiers = np.load(tiers_path)

        if ds == "non_train":
            if tiers.dtype.kind in ("U", "S") or tiers.dtype == object:
                indices = np.where(tiers != "train")[0].tolist()
            else:
                indices = np.where(tiers != 0)[0].tolist()
        elif tiers.dtype.kind in ("U", "S") or tiers.dtype == object:
            tier_map = {"train": "train", "val": "validation",
                        "validation": "validation", "test": "test"}
            split_str = tier_map.get(ds, data_split)
            if split_str == "validation":
                indices = np.where((tiers == "validation") | (tiers == "val"))[0].tolist()
            else:
                indices = np.where(tiers == split_str)[0].tolist()
        else:
            tier_val = {"train": 0, "val": 1, "validation": 1, "test": 2}.get(ds, data_split)
            indices = np.where(tiers == tier_val)[0].tolist()

        valid = []
        for i in indices:
            if (os.path.isfile(os.path.join(data_videos, f"{i}.npy"))
                    and os.path.isfile(os.path.join(data_responses, f"{i}.npy"))):
                valid.append(i)
            else:
                self._missing.append(i)
        return valid

    def _load_response_std(self):
        """Load per-neuron response std, averaged across time bins.

        std.npy is (N_neurons, T_bins); we average over axis=1 (time) to get
        one std value per neuron.
        """
        std_path = os.path.join(
            self.root_dir, "meta", "statistics", "responses", "all", "std.npy"
        )
        if not os.path.isfile(std_path):
            return None
        std_arr = np.load(std_path)
        if std_arr.ndim > 1:
            squeezed = np.squeeze(std_arr)
            if squeezed.ndim <= 1:
                std_out = squeezed.astype(np.float64)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    std_out = np.nanmean(std_arr, axis=1).astype(np.float64)
        else:
            std_out = np.squeeze(std_arr).astype(np.float64)
        return np.where(std_out > 1e-12, std_out, 1.0)

    # ------------------------------------------------------------------
    def __len__(self):
        n = len(self.trial_indices)
        if self.vid_frame == "per_frame" and self._n_frames is not None:
            return n * self._n_frames
        return n

    @property
    def num_neurons(self):
        return self._num_neurons

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _load_behavior(self, trial_id):
        """Return 6-dim behavior vector (or per-frame array)."""
        root = self.root_dir
        behav_path = os.path.join(root, "data", "behavior", f"{trial_id}.npy")
        pupil_path = os.path.join(root, "data", "pupil_center", f"{trial_id}.npy")
        if os.path.isfile(behav_path):
            b = np.load(behav_path)
        else:
            b = np.zeros(2)
        if os.path.isfile(pupil_path):
            p = np.load(pupil_path)
        else:
            p = np.zeros(2)
        return b, p

    def _behav_to_6dim(self, b_raw, p_raw):
        """Collapse raw behavior/pupil arrays to a single 6-dim vector."""
        if b_raw.ndim > 1:
            b = np.nanmean(b_raw, axis=-1).flatten()
        else:
            b = b_raw.flatten()
        running_speed = float(b[1]) if b.size >= 2 else 0.0
        pupil_dilation = float(b[0]) if b.size >= 2 else 0.0

        if p_raw.ndim > 1:
            p = np.nanmean(p_raw, axis=-1).flatten()
        else:
            p = p_raw.flatten()
        pupil_x = float(p[0]) if p.size > 0 else 0.0
        pupil_y = float(p[1]) if p.size > 1 else 0.0

        v = np.array([running_speed, 0.0, 0.0, pupil_x, pupil_y, pupil_dilation],
                      dtype=np.float32)
        return np.nan_to_num(v)

    def _standardize_resp(self, resp):
        resp = resp.astype(np.float32)
        if self.standardize_responses and self._response_std is not None:
            resp = resp / self._response_std
        resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(resp, 0.0)

    # ------------------------------------------------------------------
    # __getitem__ dispatches to the right mode
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        if self.vid_frame == "per_frame":
            return self._getitem_per_frame(idx)
        return self._getitem_trial_mean(idx)

    # ---------- original: one sample per trial ----------
    def _getitem_trial_mean(self, idx):
        trial_id = self.trial_indices[idx]
        root = self.root_dir

        vid = np.load(os.path.join(root, "data", "videos", f"{trial_id}.npy"))
        if vid.ndim == 3:
            frame = np.nanmean(vid, axis=0) if self.vid_frame == "mean" else vid[vid.shape[0] // 2]
        else:
            frame = np.asarray(vid)
        frame = resize_video_frame(frame, self.target_size)
        frame = normalize_movie(frame)
        frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
        frame = np.clip(frame, 0.0, 1.0)
        image = torch.tensor(frame[np.newaxis, np.newaxis], dtype=torch.float32)

        b_raw, p_raw = self._load_behavior(trial_id)
        behav_6 = self._behav_to_6dim(b_raw, p_raw)
        behav = torch.tensor(behav_6.reshape(1, -1).repeat(self.seq_len, axis=0),
                             dtype=torch.float32)

        resp = np.load(os.path.join(root, "data", "responses", f"{trial_id}.npy"))
        if resp.ndim > 1:
            resp = np.nanmean(resp, axis=1)  # (N_neurons, T_bins) → (N_neurons,)
        spikes = torch.tensor(self._standardize_resp(resp), dtype=torch.float32)

        return image, behav, spikes

    # ---------- per_frame: one sample per (trial, frame) ----------
    def _getitem_per_frame(self, idx):
        n_frames = self._n_frames
        trial_pos = idx // n_frames
        frame_idx = idx % n_frames
        trial_id = self.trial_indices[trial_pos]
        root = self.root_dir

        # --- video frame ---
        vid = np.load(os.path.join(root, "data", "videos", f"{trial_id}.npy"))
        frame = vid[frame_idx] if vid.ndim == 3 else np.asarray(vid)
        frame = resize_video_frame(frame, self.target_size)
        frame = normalize_movie(frame)
        frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
        frame = np.clip(frame, 0.0, 1.0)
        image = torch.tensor(frame[np.newaxis, np.newaxis], dtype=torch.float32)

        # --- behavior binned to this frame ---
        b_raw, p_raw = self._load_behavior(trial_id)
        if b_raw.ndim > 1 and b_raw.shape[-1] > 1:
            b_binned = _bin_behavior_to_frames(
                b_raw if b_raw.shape[0] < b_raw.shape[-1] else b_raw.T,
                n_frames,
            )
            running_speed = float(b_binned[frame_idx, 1]) if b_binned.shape[1] >= 2 else 0.0
            pupil_dilation = float(b_binned[frame_idx, 0]) if b_binned.shape[1] >= 2 else 0.0
        else:
            b = np.nanmean(b_raw, axis=-1).flatten() if b_raw.ndim > 1 else b_raw.flatten()
            running_speed = float(b[1]) if b.size >= 2 else 0.0
            pupil_dilation = float(b[0]) if b.size >= 2 else 0.0

        if p_raw.ndim > 1 and p_raw.shape[-1] > 1:
            p_binned = _bin_behavior_to_frames(
                p_raw if p_raw.shape[0] < p_raw.shape[-1] else p_raw.T,
                n_frames,
            )
            pupil_x = float(p_binned[frame_idx, 0]) if p_binned.shape[1] > 0 else 0.0
            pupil_y = float(p_binned[frame_idx, 1]) if p_binned.shape[1] > 1 else 0.0
        else:
            p = np.nanmean(p_raw, axis=-1).flatten() if p_raw.ndim > 1 else p_raw.flatten()
            pupil_x = float(p[0]) if p.size > 0 else 0.0
            pupil_y = float(p[1]) if p.size > 1 else 0.0

        behav_6 = np.nan_to_num(np.array(
            [running_speed, 0.0, 0.0, pupil_x, pupil_y, pupil_dilation], dtype=np.float32,
        ))
        behav = torch.tensor(behav_6.reshape(1, -1).repeat(self.seq_len, axis=0),
                             dtype=torch.float32)

        # --- responses binned to this frame ---
        # Raw shape: (N_neurons, T_bins); transpose to (T_bins, N_neurons) for binning
        resp = np.load(os.path.join(root, "data", "responses", f"{trial_id}.npy"))
        if resp.ndim > 1 and resp.shape[1] > 1:
            binned = _bin_responses_to_frames(resp.T, n_frames)  # (n_frames, N_neurons)
            resp_frame = binned[frame_idx]
        else:
            resp_frame = resp.flatten() if resp.ndim <= 1 else np.nanmean(resp, axis=1)
        spikes = torch.tensor(self._standardize_resp(resp_frame), dtype=torch.float32)

        return image, behav, spikes
