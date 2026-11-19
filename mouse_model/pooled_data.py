"""Helpers for pooled multi-session / multi-animal CNN training (padding + splits)."""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from mouse_model.data_utils_new import MouseDatasetSegNewBehav
from mouse_model.sensorium_cache import SensoriumCachedDataset, build_sensorium_cache
from mouse_model.splits import mouse_time_block_split, sensorium_trial_split


class SessionTagDataset(Dataset):
    """Wraps a (image, behav, spikes) dataset and appends an integer session_id."""

    def __init__(self, base: Dataset, session_id: int):
        self.base = base
        self.session_id = int(session_id)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, behav, spikes = self.base[idx]
        return img, behav, spikes, self.session_id


def collate_padded_multi(batch, max_neurons: int):
    """Stack batch; pad spike vectors to max_neurons (valid targets live in first n columns per session)."""
    imgs = torch.stack([b[0] for b in batch])
    behav = torch.stack([b[1] for b in batch])
    sids = torch.tensor([b[3] for b in batch], dtype=torch.long)
    bsz = len(batch)
    spikes = torch.zeros(bsz, max_neurons, dtype=torch.float32)
    for i, b in enumerate(batch):
        s = b[2]
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float32)
        s = s.reshape(-1).to(dtype=torch.float32)
        n = min(s.shape[0], max_neurons)
        spikes[i, :n] = s[:n]
    return imgs, behav, spikes, sids


def _per_segment_train_val(
    ds: Dataset, train_ratio_m: float, rng: np.random.RandomState
):
    n = len(ds)
    n_train = int(n * train_ratio_m)
    perm = rng.permutation(n)
    tr = perm[:n_train].tolist()
    va = perm[n_train:].tolist()
    return Subset(ds, tr), Subset(ds, va)


def build_pooled_mouse_datasets(
    file_ids: list[str],
    *,
    segment_num: int,
    vid_type: str,
    seq_len: int,
    train_ratio_m: float = 0.7,
    seed: int = 0,
    max_train_samples: int | None = None,
    split: str = "time_blocks",
    block_len: int = 250,
    gap: int | None = None,
):
    """
    Returns train_ds, val_ds, readout_dims (str keys "0","1",...), collate_fn, meta.

    split="time_blocks" (default): each mouse uses mouse_time_block_split (whole ~12 s blocks
    go to train or val, with a gap). max_train_samples is divided equally across mice.
    split="random_frames": the earlier per-segment random permutation (adjacent frames leak).
    """
    if split == "time_blocks":
        tagged_tr, tagged_va = [], []
        readout_dims: dict[str, int] = {}
        split_infos = []
        per_mouse_cap = None if max_train_samples is None else max_train_samples // len(file_ids)
        for sid, fid in enumerate(file_ids):
            tr, va, info = mouse_time_block_split(
                fid, segment_num=segment_num, vid_type=vid_type, seq_len=seq_len,
                block_len=block_len, val_frac=1.0 - train_ratio_m, gap=gap, seed=seed + sid,
                max_train_samples=per_mouse_cap,
            )
            readout_dims[str(sid)] = info["num_neurons"]
            tagged_tr.append(SessionTagDataset(tr, sid))
            tagged_va.append(SessionTagDataset(va, sid))
            split_infos.append(info)
        max_n = max(readout_dims.values())

        def collate_fn(batch):
            return collate_padded_multi(batch, max_n)

        meta = {
            "kind": "mouse_pooled",
            "file_ids": list(file_ids),
            "readout_dims": readout_dims,
            "max_neurons": max_n,
            "split": "mouse_time_blocks",
            "split_infos": split_infos,
            "train_minutes": sum(i["train_minutes"] for i in split_infos),
        }
        return ConcatDataset(tagged_tr), ConcatDataset(tagged_va), readout_dims, collate_fn, meta
    if split != "random_frames":
        raise ValueError("split must be 'time_blocks' or 'random_frames'")

    rng = np.random.RandomState(seed)
    tagged_tr, tagged_va = [], []
    readout_dims: dict[str, int] = {}

    for sid, fid in enumerate(file_ids):
        ds_list = [
            MouseDatasetSegNewBehav(
                file_id=fid,
                segment_num=segment_num,
                seg_idx=i,
                data_split="train",
                vid_type=vid_type,
                seq_len=seq_len,
                predict_offset=1,
            )
            for i in range(segment_num)
        ]
        tr_parts, va_parts = [], []
        for seg_ds in ds_list:
            tr, va = _per_segment_train_val(seg_ds, train_ratio_m, rng)
            tr_parts.append(tr)
            va_parts.append(va)
        mouse_tr = ConcatDataset(tr_parts)
        mouse_va = ConcatDataset(va_parts)
        probe = MouseDatasetSegNewBehav(
            file_id=fid,
            segment_num=segment_num,
            seg_idx=0,
            data_split="train",
            vid_type=vid_type,
            seq_len=seq_len,
            predict_offset=1,
        )
        readout_dims[str(sid)] = int(probe.nsp.shape[1])
        tagged_tr.append(SessionTagDataset(mouse_tr, sid))
        tagged_va.append(SessionTagDataset(mouse_va, sid))

    train_ds = ConcatDataset(tagged_tr)
    val_ds = ConcatDataset(tagged_va)
    max_n = max(readout_dims.values())

    if max_train_samples is not None and len(train_ds) > max_train_samples:
        perm = np.random.RandomState(seed).permutation(len(train_ds))[:max_train_samples]
        train_ds = Subset(train_ds, perm.tolist())

    def collate_fn(batch):
        return collate_padded_multi(batch, max_n)

    meta = {
        "kind": "mouse_pooled",
        "file_ids": list(file_ids),
        "readout_dims": readout_dims,
        "max_neurons": max_n,
        "split": "random_frames",
    }
    return train_ds, val_ds, readout_dims, collate_fn, meta


def get_or_create_concat_split(
    concat_ds: ConcatDataset,
    lengths: list[int],
    split_tag: str,
    train_ratio: float,
    seed: int,
    cache_dir: str,
):
    """Random split over flattened ConcatDataset indices; cache pickle under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    ratio_str = str(int(train_ratio * 100))
    path = os.path.join(
        cache_dir,
        f"split_concat_{ratio_str}_{100 - int(train_ratio * 100)}_{split_tag}.pkl",
    )
    n = len(concat_ds)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            sp = pickle.load(f)
        if (
            sp.get("n_total") == n
            and sp.get("lengths") == lengths
            and "train_indices" in sp
            and "val_indices" in sp
        ):
            print(f"Loaded concat split from {path}")
            return sp["train_indices"], sp["val_indices"]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()
    n_train = int(n * train_ratio)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    with open(path, "wb") as f:
        pickle.dump(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "seed": seed,
                "train_ratio": train_ratio,
                "tag": split_tag,
                "lengths": lengths,
                "n_total": n,
            },
            f,
        )
    print(f"Saved concat split to {path}")
    return train_indices, val_indices


def validate_sensorium_roots(root_dirs: list[str]) -> None:
    """Fail fast with a clear message if any root is missing Sensorium layout."""
    missing = []
    for i, root in enumerate(root_dirs):
        root = os.path.expanduser(root)
        videos = os.path.join(root, "data", "videos")
        responses = os.path.join(root, "data", "responses")
        if not os.path.isdir(videos):
            missing.append(f"  [{i}] root={root!r} -> missing {videos}")
        elif not os.path.isdir(responses):
            missing.append(f"  [{i}] root={root!r} -> missing {responses}")
    if missing:
        raise FileNotFoundError(
            "Sensorium dataset root(s) invalid. Each root must contain "
            "data/videos/ and data/responses/ (unzipped competition export).\n"
            + "\n".join(missing)
        )


def build_pooled_sensorium_datasets(
    root_dirs: list[str],
    *,
    seq_len: int = 1,
    vid_frame: str = "per_frame",
    train_ratio: float = 0.7,
    seed: int = 0,
    split_cache_dir: str | None = None,
    split_tag: str = "pooled_sensorium",
    response_lag: int = 1,
    max_train_samples: int | None = None,
):
    """
    One SensoriumCachedDataset per root (cache built on first use), each split by whole
    trial over all trials (competition tiers ignored). max_train_samples is divided
    equally across sessions. vid_frame / split_cache_dir / split_tag are accepted for
    compatibility with older notebooks and ignored.
    """
    if seq_len != 1:
        raise ValueError("Cached Sensorium samples are single frames (seq_len=1)")
    validate_sensorium_roots(root_dirs)
    tagged_tr, tagged_va = [], []
    readout_dims: dict[str, int] = {}
    split_infos = []
    per_session_cap = None if max_train_samples is None else max_train_samples // len(root_dirs)
    for sid, root in enumerate(root_dirs):
        ds = SensoriumCachedDataset(build_sensorium_cache(root), response_lag=response_lag)
        tr, va, info = sensorium_trial_split(ds, val_frac=1.0 - train_ratio, seed=seed + sid,
                                             max_train_samples=per_session_cap)
        info["root_dir"] = root
        readout_dims[str(sid)] = int(ds.num_neurons)
        tagged_tr.append(SessionTagDataset(tr, sid))
        tagged_va.append(SessionTagDataset(va, sid))
        split_infos.append(info)
    max_n = max(readout_dims.values())

    def collate_fn(batch):
        return collate_padded_multi(batch, max_n)

    meta = {
        "kind": "sensorium_pooled",
        "root_dirs": list(root_dirs),
        "readout_dims": readout_dims,
        "max_neurons": max_n,
        "split": "sensorium_by_trial",
        "response_lag": response_lag,
        "split_infos": split_infos,
        "train_minutes": sum(i["train_minutes"] for i in split_infos),
    }
    return ConcatDataset(tagged_tr), ConcatDataset(tagged_va), readout_dims, collate_fn, meta
