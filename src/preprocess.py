
import json, shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io_utils import load_any, safe_name
from .schema import canonicalize
from .text_norm import normalize_text
from .splits import stratified_split_indices, stratified_split_indices_multi, save_json

KEEP_COLS = ["row_id","task","label","variety_name","source_name","text","text_norm"]

def find_raw_files(raw_dir: Path) -> Tuple[Path, Path]:
    candidates = list(raw_dir.glob("besstie_train.*"))
    candidates2 = list(raw_dir.glob("besstie_validation.*"))
    if len(candidates) != 1 or len(candidates2) != 1:
        raise FileNotFoundError(f"Missing train/val in {raw_dir}")
    return candidates[0], candidates2[0]

def summarize(df: pd.DataFrame) -> Dict:
    return {
        "n": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict() if len(df) else {},
        "variety_counts": df["variety_name"].value_counts().to_dict() if len(df) else {},
        "source_counts": df["source_name"].value_counts().to_dict() if len(df) else {},
    }

def filter_trainpool(df_task_train: pd.DataFrame, task: str, setting: str) -> pd.DataFrame:
    # Logic: Sentiment (Source or Variety); Sarcasm (FULL or Variety)
    if task == "sentiment":
        if setting in df_task_train["source_name"].unique():
            return df_task_train[df_task_train["source_name"] == setting].copy()
        if setting.startswith("TRAIN_"):
            v = setting.replace("TRAIN_", "")
            return df_task_train[df_task_train["variety_name"] == v].copy()
        return df_task_train.copy()

    if task == "sarcasm":
        if setting == "FULL":
            return df_task_train.copy()
        if setting.startswith("TRAIN_"):
            v = setting.replace("TRAIN_", "")
            return df_task_train[df_task_train["variety_name"] == v].copy()

    return df_task_train.copy()

def build_testsets(df_task_valid: pd.DataFrame, out_dir_task: Path) -> pd.DataFrame:
    # Writes to proc_dir/<task>/testsets/
    test_dir = out_dir_task / "testsets"
    test_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # 1. FULL
    p_full = test_dir / "TEST_FULL.csv"
    df_task_valid[KEEP_COLS].to_csv(p_full, index=False)
    rows.append({"test_setting":"TEST_FULL", "csv":str(p_full), **summarize(df_task_valid)})

    # 2. By Source
    for src in sorted(df_task_valid["source_name"].unique()):
        df_s = df_task_valid[df_task_valid["source_name"] == src].copy()
        p = test_dir / f"TEST_{safe_name(src)}.csv"
        df_s[KEEP_COLS].to_csv(p, index=False)
        rows.append({"test_setting":f"TEST_{src}", "csv":str(p), **summarize(df_s)})

    # 3. By Variety
    for v in sorted(df_task_valid["variety_name"].unique()):
        df_v = df_task_valid[df_task_valid["variety_name"] == v].copy()
        p = test_dir / f"TEST_{safe_name(v)}.csv"
        df_v[KEEP_COLS].to_csv(p, index=False)
        rows.append({"test_setting":f"TEST_{v}", "csv":str(p), **summarize(df_v)})

    idx = pd.DataFrame(rows).sort_values("test_setting").reset_index(drop=True)
    idx.to_csv(test_dir / "index_testsets.csv", index=False)
    return idx

def run_preprocess(raw_dir, proc_dir, splits_dir, report_dir, seed, val_ratio, sarc_source_only, max_len_for_models):
    train_file, valid_file = find_raw_files(raw_dir)

    # Load & Canonicalize
    df_train = canonicalize(load_any(train_file))
    df_valid = canonicalize(load_any(valid_file))

    # Normalize Text
    df_train["text_norm"] = df_train["text"].apply(normalize_text)
    df_valid["text_norm"] = df_valid["text"].apply(normalize_text)

    # IDs
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_train["row_id"] = np.arange(len(df_train), dtype=np.int64)
    df_valid["row_id"] = np.arange(len(df_valid), dtype=np.int64) + int(10**9)

    # Filter Sarcasm Source (Handle Case Insensitivity via Canonical Names)
    # Note: canonicalize() now forces "Reddit" to be Title Case, so simple comparison is safe.
    mask_tr_sarc = df_train["task"] == "sarcasm"
    mask_va_sarc = df_valid["task"] == "sarcasm"

    if mask_tr_sarc.any():
        # sarc_source_only from config is usually "Reddit"
        df_train = pd.concat([df_train[~mask_tr_sarc], df_train[mask_tr_sarc & (df_train["source_name"] == sarc_source_only)]], ignore_index=True)
    if mask_va_sarc.any():
        df_valid = pd.concat([df_valid[~mask_va_sarc], df_valid[mask_va_sarc & (df_valid["source_name"] == sarc_source_only)]], ignore_index=True)

    # Reset IDs after filtering
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_train["row_id"] = np.arange(len(df_train), dtype=np.int64)
    df_valid["row_id"] = np.arange(len(df_valid), dtype=np.int64) + int(10**9)

    # --- SAVE FULL SNAPSHOTS ---
    (proc_dir / "_all").mkdir(parents=True, exist_ok=True)
    df_train[KEEP_COLS].to_csv(proc_dir / "_all" / "train_all.csv", index=False)
    df_valid[KEEP_COLS].to_csv(proc_dir / "_all" / "validation_all.csv", index=False)

    tasks = sorted(df_train["task"].unique().tolist())
    all_index_rows = []
    audit_rows = []

    for task in tasks:
        # Create Task Directory Structure
        task_dir = proc_dir / safe_name(task)
        trainsets_dir = task_dir / "trainsets"
        splits_out_dir = task_dir / "splits"

        for d in [task_dir, trainsets_dir, splits_out_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Build Testsets
        df_task_valid = df_valid[df_valid["task"] == task].copy()
        testsets_index_path = task_dir / "testsets" / "index_testsets.csv"
        if len(df_task_valid) > 0:
            build_testsets(df_task_valid, task_dir)

        # 2. Build Training Settings
        df_task_train = df_train[df_train["task"] == task].copy()
        if len(df_task_train) == 0: continue

        # Define Settings
        if task == "sentiment":
            settings = sorted(df_task_train["source_name"].unique().tolist()) + \
                       [f"TRAIN_{v}" for v in sorted(df_task_train["variety_name"].unique())]
        elif task == "sarcasm":
            settings = ["FULL"] + [f"TRAIN_{v}" for v in sorted(df_task_train["variety_name"].unique())]
        else:
            settings = ["FULL"]

        task_settings_rows = []

        for setting in settings:
            tr_pool = filter_trainpool(df_task_train, task, setting)
            if len(tr_pool) < 10: continue

            # --- Fix B: IMPROVED STRATIFICATION ---
            n_var = tr_pool["variety_name"].nunique()
            n_src = tr_pool["source_name"].nunique()

            # If variety differs, split by label+variety
            if n_var > 1:
                strategy = "label_variety"
                strata = (tr_pool["label"].astype(str) + "__" + tr_pool["variety_name"]).values
                tr_idx, va_idx = stratified_split_indices_multi(strata, val_ratio, seed)
            # If variety is fixed (e.g. TRAIN_Au) but source differs (Google/Reddit), split by label+source
            elif n_src > 1:
                strategy = "label_source"
                strata = (tr_pool["label"].astype(str) + "__" + tr_pool["source_name"]).values
                tr_idx, va_idx = stratified_split_indices_multi(strata, val_ratio, seed)
            # Otherwise simple stratified
            else:
                strategy = "label"
                strata = tr_pool["label"].values
                tr_idx, va_idx = stratified_split_indices(strata, val_ratio, seed)

            df_tr = tr_pool.iloc[tr_idx][KEEP_COLS].reset_index(drop=True)
            df_va = tr_pool.iloc[va_idx][KEEP_COLS].reset_index(drop=True)

            out_dir = trainsets_dir / safe_name(setting)
            out_dir.mkdir(parents=True, exist_ok=True)

            p_tr = out_dir / "train.csv"
            p_va = out_dir / "val.csv"
            df_tr.to_csv(p_tr, index=False)
            df_va.to_csv(p_va, index=False)

            # Splits JSON
            split_obj = {
                "task": task, "setting": setting, "seed": seed, "val_ratio": val_ratio,
                "split_strategy": strategy, "max_len_for_models": int(max_len_for_models),
                "train_row_ids": df_tr["row_id"].tolist(), "val_row_ids": df_va["row_id"].tolist(),
            }
            p_split_global = splits_dir / f"{safe_name(task)}__{safe_name(setting)}.json"
            save_json(p_split_global, split_obj)

            p_split_local = splits_out_dir / f"{safe_name(setting)}.json"
            save_json(p_split_local, split_obj)

            # Manifest
            manifest = {
                "task": task, "setting": setting,
                "files": {"train_csv": str(p_tr), "val_csv": str(p_va)},
                "splits_json": str(p_split_global),
                "testsets_index_csv": str(testsets_index_path),
                "summary": {"train": summarize(df_tr), "val": summarize(df_va)}
            }
            save_json(out_dir / "manifest.json", manifest)

            # Record for indices
            row = {
                "task": task, "setting": setting,
                "train_csv": str(p_tr), "val_csv": str(p_va),
                "splits_json": str(p_split_global),
                "testsets_index_csv": str(testsets_index_path),
                "n_train": len(df_tr), "n_val": len(df_va)
            }
            all_index_rows.append(row)
            task_settings_rows.append(row)

            # Audit
            audit_rows.append({
                "task": task, "setting": setting, "split_strategy": strategy,
                "train_label_dist": dict(df_tr["label"].value_counts(normalize=True)),
                "val_label_dist": dict(df_va["label"].value_counts(normalize=True))
            })

        # --- SAVE PER-TASK INDICES ---
        if task_settings_rows:
            df_ts = pd.DataFrame(task_settings_rows)
            # 1. processed/<task>/index_settings.csv
            df_ts.to_csv(task_dir / "index_settings.csv", index=False)
            # 2. processed/<task>/trainsets/index_trainsets.csv
            df_ts.to_csv(trainsets_dir / "index_trainsets.csv", index=False)

            # 3. processed/<task>/train/index_train.csv (Fake this one if loader looks for it)
            (task_dir / "train").mkdir(exist_ok=True)
            df_ts.to_csv(task_dir / "train" / "index_train.csv", index=False)

            # 4. Fix D: processed/<task>/index_train.csv (The missing one!)
            df_ts.to_csv(task_dir / "index_train.csv", index=False)

    # Global Index
    index_df = pd.DataFrame(all_index_rows)
    if not index_df.empty:
        index_df = index_df.sort_values(["task", "setting"]).reset_index(drop=True)
        index_df.to_csv(proc_dir / "index.csv", index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(report_dir / "preprocess_audit.csv", index=False)

    return index_df
