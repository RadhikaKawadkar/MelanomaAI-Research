"""
split_dataset.py — Leak-Free Clinical Dataset Partitioning
=========================================================
Ensures valid data splitting for MelanomaAI:
1. Patient / Lesion Grouping: Multiple images of the same lesion (or patient)
   are strictly confined to a single partition (never split across train & test).
2. Synthetic Image Exclusion: Offline augmented images (e.g. 'aug_*') are
   strictly restricted to the training partition and banned from test/eval.
3. Three-Way Split: Train (model fitting), Tuning/Calibration (threshold & Platt scaling),
   and Test (final independent unbiased evaluation).
"""

from __future__ import annotations
import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def hash_file(filepath: Path) -> str:
    """Returns SHA-256 hash of image file to detect duplicate images."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def partition_dataset_grouped(
    metadata_df: pd.DataFrame,
    group_col: str = "lesion_id",
    label_col: str = "dx",
    positive_label: str = "mel",
    train_ratio: float = 0.60,
    calib_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Partitions a dataset metadata DataFrame grouped by lesion or patient ID.
    Guarantees 0% lesion overlap between train, calibration, and test sets.
    """
    assert abs((train_ratio + calib_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    if group_col not in metadata_df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in metadata.")

    # Unique groups with their primary class label for stratified group splitting
    group_summary = metadata_df.groupby(group_col)[label_col].agg(
        lambda x: (x == positive_label).mean()
    ).reset_index()
    group_summary["is_positive"] = (group_summary[label_col] > 0.0).astype(int)

    train_groups = []
    calib_groups = []
    test_groups = []

    for is_pos in [0, 1]:
        subset = group_summary[group_summary["is_positive"] == is_pos][group_col].values
        rng.shuffle(subset)
        n = len(subset)
        n_train = int(train_ratio * n)
        n_calib = int(calib_ratio * n)

        train_groups.extend(subset[:n_train])
        calib_groups.extend(subset[n_train: n_train + n_calib])
        test_groups.extend(subset[n_train + n_calib:])

    train_df = metadata_df[metadata_df[group_col].isin(set(train_groups))].copy()
    calib_df = metadata_df[metadata_df[group_col].isin(set(calib_groups))].copy()
    test_df = metadata_df[metadata_df[group_col].isin(set(test_groups))].copy()

    # Integrity verification
    train_g = set(train_df[group_col])
    calib_g = set(calib_df[group_col])
    test_g = set(test_df[group_col])

    overlap_tc = train_g.intersection(calib_g)
    overlap_tt = train_g.intersection(test_g)
    overlap_ct = calib_g.intersection(test_g)

    if overlap_tc or overlap_tt or overlap_ct:
        raise RuntimeError("Data leakage detected: Group overlap between partitions!")

    return {
        "train": train_df,
        "calibration": calib_df,
        "test": test_df,
    }


def filter_leak_free_images(
    image_paths: List[Path],
    is_test_or_val: bool = True
) -> List[Path]:
    """
    Excludes synthetic augmented images from test/validation evaluation sets.
    """
    if not is_test_or_val:
        return image_paths
    clean_paths = [
        p for p in image_paths
        if not p.name.startswith("aug_") and not "_aug" in p.name.lower()
    ]
    return clean_paths
