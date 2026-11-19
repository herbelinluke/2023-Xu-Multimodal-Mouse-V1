"""
Leak-resistant train/val splits.

- Sensorium: whole trials go to train or val (all trials pooled; competition tiers are ignored).
- Mouse: each segment is cut into contiguous time blocks; whole blocks go to train or val,
  and `gap` samples on either side of every val block are dropped from train.

Each split also returns a per-sample group id for the training set, so learning-curve
subsets can be built from whole trials/blocks with limit_by_groups().
"""
from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from mouse_model.data_utils_new import MouseDatasetSegNewBehav

MOUSE_FRAME_S = 0.048


def split_hash(train_idx, val_idx) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    h.update(b"|")
    h.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:12]


def split_by_groups(groups: np.ndarray, val_frac: float, seed: int):
    """Assign whole groups to val (fraction of groups) and the rest to train."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(val_frac * len(uniq))))
    val_groups = rng.permutation(uniq)[:n_val]
    is_val = np.isin(groups, val_groups)
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def limit_by_groups(train_idx: np.ndarray, train_groups: np.ndarray, max_samples: int | None, seed: int):
    """Take whole training groups in random order until max_samples; the last group is truncated."""
    train_idx = np.asarray(train_idx)
    if max_samples is None or max_samples >= len(train_idx):
        return train_idx
    rng = np.random.RandomState(seed)
    order = rng.permutation(np.unique(train_groups))
    picked, total = [], 0
    for g in order:
        members = train_idx[train_groups == g]
        take = min(len(members), max_samples - total)
        picked.append(members[:take])
        total += take
        if total >= max_samples:
            break
    return np.concatenate(picked)


def sensorium_trial_split(ds, *, val_frac: float = 0.3, seed: int = 0, max_train_samples: int | None = None):
    """ds: SensoriumCachedDataset. Returns train_ds, val_ds, info."""
    groups = ds.trial_of_sample
    tr, va = split_by_groups(groups, val_frac, seed)
    tr_groups = groups[tr]
    tr = limit_by_groups(tr, tr_groups, max_train_samples, seed)
    info = {
        "split": "sensorium_by_trial",
        "val_frac": val_frac,
        "seed": seed,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_train_groups": int(len(np.unique(groups[tr]))),
        "n_val_groups": int(len(np.unique(groups[va]))),
        "train_minutes": len(tr) / ds.fps / 60.0,
        "val_minutes": len(va) / ds.fps / 60.0,
        "max_train_samples": max_train_samples,
        "split_hash": split_hash(tr, va),
    }
    return Subset(ds, tr.tolist()), Subset(ds, va.tolist()), info


def _block_split_one_segment(n: int, block_len: int, val_frac: float, gap: int, rng):
    block = np.arange(n) // block_len
    n_blocks = int(block.max()) + 1
    n_val = min(max(1, int(round(val_frac * n_blocks))), n_blocks - 1)
    val_blocks = rng.permutation(n_blocks)[:n_val]
    is_val = np.isin(block, val_blocks)
    near_val = is_val.copy()
    for s in range(1, gap + 1):
        near_val[s:] |= is_val[:-s]
        near_val[:-s] |= is_val[s:]
    train = np.nonzero(~near_val)[0]
    val = np.nonzero(is_val)[0]
    return train, val, block


def mouse_time_block_split(
    file_id: str,
    *,
    segment_num: int = 10,
    vid_type: str = "vid_mean",
    seq_len: int = 1,
    predict_offset: int = 1,
    block_len: int = 250,
    val_frac: float = 0.3,
    gap: int | None = None,
    seed: int = 0,
    max_train_samples: int | None = None,
    data_split: str = "train",
):
    """
    Returns train_ds, val_ds, info. block_len is in samples (250 x 48 ms = 12 s).
    gap defaults to seq_len + predict_offset so no train sample shares a frame or target with val.
    """
    gap = seq_len + predict_offset if gap is None else gap
    rng = np.random.RandomState(seed)
    seg_ds = [
        MouseDatasetSegNewBehav(file_id=file_id, segment_num=segment_num, seg_idx=i, data_split=data_split,
                                vid_type=vid_type, seq_len=seq_len, predict_offset=predict_offset)
        for i in range(segment_num)
    ]
    full = ConcatDataset(seg_ds)
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in seg_ds])[:-1]])

    tr_all, va_all, groups_all = [], [], np.zeros(len(full), dtype=np.int64)
    group_base = 0
    for d, off in zip(seg_ds, offsets):
        tr, va, block = _block_split_one_segment(len(d), block_len, val_frac, gap, rng)
        tr_all.append(tr + off)
        va_all.append(va + off)
        groups_all[off:off + len(d)] = block + group_base
        group_base += int(block.max()) + 1
    tr = np.concatenate(tr_all)
    va = np.concatenate(va_all)
    tr = limit_by_groups(tr, groups_all[tr], max_train_samples, seed)

    info = {
        "split": "mouse_time_blocks",
        "file_id": file_id,
        "segment_num": segment_num,
        "block_len": block_len,
        "block_seconds": block_len * MOUSE_FRAME_S,
        "gap": gap,
        "val_frac": val_frac,
        "seed": seed,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_train_groups": int(len(np.unique(groups_all[tr]))),
        "n_val_groups": int(len(np.unique(groups_all[va]))),
        "train_minutes": len(tr) * MOUSE_FRAME_S / 60.0,
        "val_minutes": len(va) * MOUSE_FRAME_S / 60.0,
        "max_train_samples": max_train_samples,
        "num_neurons": int(seg_ds[0].nsp.shape[1]),
        "split_hash": split_hash(tr, va),
    }
    return Subset(full, tr.tolist()), Subset(full, va.tolist()), info
