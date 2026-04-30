from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from .aug import ECGAugment


class ECGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment=None):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int):
        x = self.x[index]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[index]


def _as_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim == 3:
        # Accept either (N, C, T) or (N, T, C). For ECG here, C is usually 1.
        if x.shape[1] > x.shape[2]:
            x = np.transpose(x, (0, 2, 1))
    else:
        raise ValueError(f"Expected x to have shape (N,T), (N,C,T), or (N,T,C), got {x.shape}")
    return x.astype(np.float32)


def _normalize_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def _normalize_minmax_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xmin = x.min(axis=-1, keepdims=True)
    xmax = x.max(axis=-1, keepdims=True)
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0


def _compute_train_stats(x_train: np.ndarray, eps: float = 1e-6):
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    return mean, std + eps


def _basic_check(x: np.ndarray, y: np.ndarray, cfg: Dict) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y have inconsistent sample counts: {x.shape[0]} vs {y.shape[0]}")
    input_length = cfg.get("input_length", None)
    if input_length is not None and int(input_length) != x.shape[-1]:
        raise ValueError(f"Expected input length {input_length}, got {x.shape[-1]}")
    n_class = int(cfg.get("n_class", len(np.unique(y))))
    if y.min() < 0 or y.max() >= n_class:
        raise ValueError(f"Labels should be in [0, {n_class - 1}], got min={y.min()}, max={y.max()}")


def load_arrays(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = cfg.get("data_dir", "data")
    x_path = f"{data_dir}/{cfg.get('x_file', 'dataset_raw.npy')}"
    y_path = f"{data_dir}/{cfg.get('y_file', 'labelset_raw.npy')}"

    x = np.load(x_path)
    y = np.load(y_path)
    x = _as_channel_first(x)
    y = y.astype(np.int64).reshape(-1)
    _basic_check(x, y, cfg)

    clip_value = cfg.get("clip_value", None)
    if clip_value is not None:
        x = np.clip(x, -float(clip_value), float(clip_value))
    return x, y


def make_holdout_split(y: np.ndarray, cfg: Dict):
    test_size = float(cfg.get("test_size", 0.2))
    val_size = float(cfg.get("val_size", 0.1))
    seed = int(cfg.get("split_seed", 0))
    stratify = y if bool(cfg.get("stratify", True)) else None

    idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
    )

    y_trainval = y[trainval_idx]
    val_ratio_inside_trainval = val_size / (1.0 - test_size)
    stratify_tv = y_trainval if bool(cfg.get("stratify", True)) else None
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_ratio_inside_trainval,
        random_state=seed,
        shuffle=True,
        stratify=stratify_tv,
    )
    info = {"split_mode": "holdout", "fold": -1, "val_fold": -1, "n_splits": -1}
    return train_idx, val_idx, test_idx, info


def make_stratified_kfold_split(y: np.ndarray, cfg: Dict):
    n_splits = int(cfg.get("n_splits", 10))
    fold = int(cfg.get("fold", 0))
    val_fold_offset = int(cfg.get("val_fold_offset", 1))
    seed = int(cfg.get("split_seed", 0))

    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold should be in [0, {n_splits - 1}], got {fold}")

    counts = np.bincount(y, minlength=int(cfg.get("n_class", len(np.unique(y)))))
    if counts.min() < n_splits:
        raise ValueError(f"n_splits={n_splits} is larger than the smallest class count {counts.min()}.")

    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in skf.split(idx, y)]

    test_fold = fold
    val_fold = (fold + val_fold_offset) % n_splits
    train_folds = [i for i in range(n_splits) if i not in (test_fold, val_fold)]

    train_idx = np.concatenate([folds[i] for i in train_folds])
    val_idx = folds[val_fold]
    test_idx = folds[test_fold]

    info = {
        "split_mode": "stratified_kfold",
        "fold": int(fold),
        "val_fold": int(val_fold),
        "n_splits": int(n_splits),
    }
    return train_idx, val_idx, test_idx, info


def make_splits(y: np.ndarray, cfg: Dict):
    split_mode = cfg.get("split_mode", "stratified_kfold").lower()
    if split_mode in ["holdout", "stratified_holdout", "random"]:
        return make_holdout_split(y, cfg)
    if split_mode in ["kfold", "stratified_kfold", "stratified-kfold"]:
        return make_stratified_kfold_split(y, cfg)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def apply_normalization(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, method: str):
    method = (method or "none").lower()
    if method == "none":
        return x_train, x_val, x_test
    if method == "zscore_sample":
        return _normalize_sample(x_train), _normalize_sample(x_val), _normalize_sample(x_test)
    if method == "minmax_sample":
        return _normalize_minmax_sample(x_train), _normalize_minmax_sample(x_val), _normalize_minmax_sample(x_test)
    if method == "zscore_train":
        mean, std = _compute_train_stats(x_train)
        return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std
    raise ValueError(f"Unknown normalization method: {method}")


def build_dataloaders(data_cfg: Dict, aug_cfg: Dict, train_cfg: Dict, fold=None):
    local_cfg = dict(data_cfg)
    if fold is not None:
        local_cfg["fold"] = int(fold)

    x, y = load_arrays(local_cfg)
    train_idx, val_idx, test_idx, split_info = make_splits(y, local_cfg)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_train, x_val, x_test = apply_normalization(
        x_train, x_val, x_test, local_cfg.get("normalize", "zscore_sample")
    )

    augment = ECGAugment(aug_cfg) if bool(aug_cfg.get("enable", False)) else None
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(local_cfg.get("num_workers", 4))
    pin_memory = bool(local_cfg.get("pin_memory", True))
    n_class = int(local_cfg.get("n_class", len(np.unique(y))))

    loaders = {
        "train": DataLoader(ECGDataset(x_train, y_train, augment=augment), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
        "val": DataLoader(ECGDataset(x_val, y_val), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(ECGDataset(x_test, y_test), batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }

    info = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_class": n_class,
        "in_channels": int(x.shape[1]),
        "input_length": int(x.shape[-1]),
        "class_counts_total": np.bincount(y, minlength=n_class).tolist(),
        "class_counts_train": np.bincount(y_train, minlength=n_class).tolist(),
        "class_counts_val": np.bincount(y_val, minlength=n_class).tolist(),
        "class_counts_test": np.bincount(y_test, minlength=n_class).tolist(),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    info.update(split_info)
    return loaders, info, y_train
