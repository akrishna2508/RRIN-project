"""
src/splits.py
=============
Patient-grouped train / validation / test splitting.

WHY THIS MATTERS (for beginners):
  If the same patient's LEFT eye is in training and RIGHT eye is in testing,
  the AI might just "recognise" the unique anatomy of that person and
  score well on the test set WITHOUT actually learning to restore images.
  This is called "data leakage" and makes evaluation results meaningless.

  We fix this by grouping ALL images from the same patient+eye and
  assigning the whole group to exactly ONE split.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.model_selection

from src.config import RANDOM_SEED, TRAIN_FRACTION, VAL_FRACTION


def assign_patient_grouped_splits(
    metadata_dataframe: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Assign each image to a train / val / test split so that no patient+eye
    combination ever appears in more than one split.

    MESSIDOR-2 images (source_dataset == 'messidor2') are ALWAYS assigned
    to "holdout" and never appear in train/val/test.

    params:
        metadata_dataframe — pandas DataFrame (must have patient_id, eye_laterality,
                             source_dataset, and is_pseudo_clean columns)
        train_fraction     — fraction of patient-eyes to use for training (default 0.85)
        val_fraction       — fraction to use for validation (default 0.10)
        random_seed        — for reproducibility
    returns: DataFrame with split_assignment column populated
    side effects: none — returns a copy, does not mutate the input

    Split assignment:
        "holdout" — MESSIDOR-2 only (never used in training)
        "train"   — training data (85% of patient-eyes)
        "val"     — validation data (10%)
        "test"    — in-domain test (remaining 5%)
    """
    df = metadata_dataframe.copy()

    # Mark MESSIDOR-2 as holdout first (they never enter the split logic)
    is_messidor = df["source_dataset"].str.lower() == "messidor2"
    df.loc[is_messidor, "split_assignment"] = "holdout"

    # Work only with non-MESSIDOR pseudo-clean images for the rest
    pseudo_clean_mask = df["is_pseudo_clean"] & (~is_messidor)
    pool = df[pseudo_clean_mask].copy()

    # Build group keys: one key per unique patient + eye combination
    group_keys = (
        pool["patient_id"].astype(str) + "_" + pool["eye_laterality"].astype(str)
    )
    unique_groups = list(group_keys.unique())

    # Shuffle groups (not images) with a fixed seed for reproducibility
    rng = random.Random(random_seed)
    rng.shuffle(unique_groups)

    # Carve out train / val / test at the GROUP level
    n_total = len(unique_groups)
    n_train = int(n_total * train_fraction)
    n_val   = int(n_total * val_fraction)

    train_groups = set(unique_groups[:n_train])
    val_groups   = set(unique_groups[n_train: n_train + n_val])
    # Remaining groups go to test

    def _assign_split(group_key: str) -> str:
        if group_key in train_groups:
            return "train"
        elif group_key in val_groups:
            return "val"
        else:
            return "test"

    df.loc[pseudo_clean_mask, "split_assignment"] = group_keys.map(_assign_split).values

    # Non-pseudo-clean, non-holdout images get "tier3_train"
    # (used in Tier-3 adversarial domain adaptation)
    tier3_mask = (~df["is_pseudo_clean"]) & (~is_messidor)
    df.loc[tier3_mask, "split_assignment"] = "tier3_train"

    _print_split_stats(df)
    return df


def _print_split_stats(df: pd.DataFrame) -> None:
    """Print a summary table of how many images are in each split."""
    counts = df["split_assignment"].value_counts()
    print("\n--- Dataset split summary ---")
    for split_name, count in counts.items():
        print(f"  {split_name:<15}: {count:>6} images")
    print()


def build_grouped_kfold_indices(
    metadata_dataframe: pd.DataFrame,
    num_folds: int = 5,
) -> list:
    """
    Generate patient-grouped k-fold indices restricted to the train+val pool.
    Used for hyperparameter search only — never for final evaluation.

    params:
        metadata_dataframe — DataFrame with split_assignment and patient_id columns
        num_folds          — number of cross-validation folds
    returns: list of (train_indices, val_indices) numpy array pairs
    side effects: none
    """
    trainval_mask = metadata_dataframe["split_assignment"].isin(["train", "val"])
    pool_df = metadata_dataframe[trainval_mask].reset_index(drop=True)

    group_keys = (
        pool_df["patient_id"].astype(str) + "_" + pool_df["eye_laterality"].astype(str)
    )

    splitter = sklearn.model_selection.GroupKFold(n_splits=num_folds)
    return list(splitter.split(pool_df, groups=group_keys))


def verify_no_patient_overlap_across_splits(
    metadata_dataframe: pd.DataFrame,
) -> bool:
    """
    Audit that no patient+eye group appears in more than one split.
    Raises AssertionError (halts the pipeline) if any overlap is found.

    This check runs automatically after every split assignment.

    params: metadata_dataframe — DataFrame with split_assignment column
    returns: True if no overlap is found (program continues)
    side effects: raises AssertionError on leakage detection
    """
    # Only check the 4 main splits (holdout doesn't interact with others)
    check_splits = ["train", "val", "test"]
    df = metadata_dataframe[metadata_dataframe["split_assignment"].isin(check_splits)]

    group_keys_by_split: dict[str, set] = {}
    for split_name in check_splits:
        subset = df[df["split_assignment"] == split_name]
        keys   = set(
            subset["patient_id"].astype(str) + "_" + subset["eye_laterality"].astype(str)
        )
        group_keys_by_split[split_name] = keys

    any_overlap = False
    for split_a, groups_a in group_keys_by_split.items():
        for split_b, groups_b in group_keys_by_split.items():
            if split_a >= split_b:
                continue
            overlap = groups_a & groups_b
            if overlap:
                print(f"  DATA LEAKAGE DETECTED between {split_a} and {split_b}!")
                print(f"  Overlapping groups: {list(overlap)[:5]} ...")
                any_overlap = True

    assert not any_overlap, (
        "Patient overlap detected across splits! "
        "This would make evaluation results unreliable. "
        "Check the splitting logic in src/splits.py."
    )

    print("Split leakage audit: PASSED — no patient/eye overlap across splits.")
    return True
