
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

def stratified_split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    y = labels.astype(int)
    
    classes = np.unique(y)
    if len(classes) < 2:
        rng.shuffle(idx)
        n_val = int(np.ceil(len(idx) * val_ratio))
        return idx[n_val:], idx[:n_val]

    train_idx, val_idx = [], []
    for c in classes:
        c_idx = idx[y == c]
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_val = int(np.round(n_c * val_ratio))
        if n_c >= 2:
            n_val = max(1, min(n_val, n_c - 1))
        else:
            n_val = 0
        val_idx.append(c_idx[:n_val])
        train_idx.append(c_idx[n_val:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx   = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def stratified_split_indices_multi(strata: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(strata))
    uniq = np.unique(strata)
    train_idx, val_idx = [], []

    for s in uniq:
        s_idx = idx[strata == s]
        rng.shuffle(s_idx)
        n_s = len(s_idx)
        n_val = int(np.round(n_s * val_ratio))
        if n_s >= 2:
            n_val = max(1, min(n_val, n_s - 1))
        else:
            n_val = 0
        val_idx.append(s_idx[:n_val])
        train_idx.append(s_idx[n_val:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx   = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
