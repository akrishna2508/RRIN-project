"""
src/dataset_parsers.py
=======================
Each public fundus dataset encodes patient ID and eye side (left/right)
differently in its filenames and CSV metadata files.

This module dispatches to the correct parser for each dataset.

ADDING A NEW DATASET:
  1. Write a function `_parse_<dataset_name>(file_path)` below.
  2. Register it in the `_PARSERS` dict at the bottom of this file.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional


# ---- EyePACS -----------------------------------------------
# Filename convention: <patient_id>_<left|right>.jpeg
# Example: 1000_left.jpeg  →  patient 1000, left eye

def _parse_eyepacs(file_path: str) -> tuple:
    stem = Path(file_path).stem                # e.g. "1000_left"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        patient_id   = parts[0]
        eye_raw      = parts[1].lower()
        laterality   = "left" if eye_raw == "left" else ("right" if eye_raw == "right" else "unknown")
    else:
        patient_id = stem
        laterality = "unknown"

    image_id = f"eyepacs_{stem}"
    return image_id, patient_id, laterality, None   # EyePACS has no per-image disease label in filename


# ---- APTOS 2019 ---------------------------------------------
# Filename: <id>.png  (integer id, no laterality in filename)
# Laterality is not recorded in APTOS, so we use "unknown".
# Disease grade is in train.csv / test.csv alongside the image folder.

_APTOS_LABELS: dict = {}   # filled lazily from CSV if found

def _load_aptos_labels(dataset_root: str) -> None:
    import pandas as pd
    for csv_name in ["train.csv", "test.csv"]:
        csv_path = os.path.join(dataset_root, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "id_code" in df.columns and "diagnosis" in df.columns:
                for _, row in df.iterrows():
                    _APTOS_LABELS[str(row["id_code"])] = str(row["diagnosis"])


def _parse_aptos(file_path: str) -> tuple:
    stem = Path(file_path).stem
    image_id = f"aptos_{stem}"
    disease_label = _APTOS_LABELS.get(stem, None)
    return image_id, stem, "unknown", disease_label


# ---- IDRiD --------------------------------------------------
# Filename: IDRiD_<number>.jpg
# Lesion annotations are in separate CSV files; we just record the number.

def _parse_idrid(file_path: str) -> tuple:
    stem = Path(file_path).stem                # e.g. "IDRiD_001"
    match = re.search(r"IDRiD_(\d+)", stem, re.IGNORECASE)
    patient_id = match.group(1) if match else stem
    image_id = f"idrid_{patient_id}"
    return image_id, patient_id, "unknown", None


# ---- MESSIDOR-2 ---------------------------------------------
# Filename: Messidor-Original/Base<nn>/<number>.tif
# This dataset is ALWAYS the held-out cross-dataset test set.
# We tag it with split_assignment="holdout" at ingest time.

def _parse_messidor2(file_path: str) -> tuple:
    stem = Path(file_path).stem
    # Try to extract the base ID from parent directory
    parent = Path(file_path).parent.name        # e.g. "Base11"
    patient_id = f"{parent}_{stem}"
    image_id   = f"messidor2_{patient_id}"
    return image_id, patient_id, "unknown", None


# ---- RFMiD --------------------------------------------------
# Filename: <integer>.png; labels in CSV

_RFMID_LABELS: dict = {}

def _load_rfmid_labels(dataset_root: str) -> None:
    import pandas as pd
    for csv_name in ["RFMiD_Training_Labels.csv", "RFMiD_Validation_Labels.csv",
                     "RFMiD_Testing_Labels.csv", "train_labels.csv"]:
        csv_path = os.path.join(dataset_root, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            id_col  = [c for c in df.columns if "id" in c.lower()][0]
            disease_col = df.columns[1]
            for _, row in df.iterrows():
                _RFMID_LABELS[str(int(row[id_col]))] = str(row[disease_col])

def _parse_rfmid(file_path: str) -> tuple:
    stem = Path(file_path).stem
    image_id = f"rfmid_{stem}"
    disease_label = _RFMID_LABELS.get(stem, None)
    return image_id, stem, "unknown", disease_label


# ---- ODIR-5K ------------------------------------------------
# Has a structured CSV with patient_id, left_fundus, right_fundus columns.

_ODIR_META: dict = {}    # filename_stem → (patient_id, laterality, label)

def _load_odir_meta(dataset_root: str) -> None:
    import pandas as pd
    for csv_name in ["ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
                     "odir_training.csv", "data.csv"]:
        csv_path = os.path.join(dataset_root, csv_name)
        if os.path.exists(csv_path):
            if csv_path.endswith(".xlsx"):
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(csv_path)
            # Expect columns: ID, Left-Fundus, Right-Fundus, Left-Diagnostic Keywords, etc.
            for _, row in df.iterrows():
                pid = str(row.get("ID", row.get("id", "")))
                for side in ["Left", "Right"]:
                    fn_col = f"{side}-Fundus"
                    lbl_col = f"{side}-Diagnostic Keywords"
                    if fn_col in row:
                        stem = Path(str(row[fn_col])).stem
                        _ODIR_META[stem] = (pid, side.lower(), row.get(lbl_col, None))
            break

def _parse_odir(file_path: str) -> tuple:
    stem = Path(file_path).stem
    if stem in _ODIR_META:
        pid, lat, lbl = _ODIR_META[stem]
    else:
        pid, lat, lbl = stem, "unknown", None
    image_id = f"odir_{stem}"
    return image_id, pid, lat, lbl


# ---- STARE / DRIVE ------------------------------------------
# Small vessel-segmentation datasets. Filenames are simple integers.

def _parse_stare(file_path: str) -> tuple:
    stem = Path(file_path).stem
    image_id = f"stare_{stem}"
    return image_id, stem, "unknown", None


def _parse_drive(file_path: str) -> tuple:
    stem = Path(file_path).stem
    # DRIVE has filenames like "21_training.tif" or "01_test.tif"
    match = re.match(r"(\d+)_(training|test)", stem)
    patient_id = match.group(1) if match else stem
    image_id = f"drive_{stem}"
    return image_id, patient_id, "unknown", None


# ---- Generic fallback ----------------------------------------

def _parse_generic(file_path: str) -> tuple:
    """
    When no specific parser exists, use a hash of the file path as the ID.
    This guarantees uniqueness even for unusual datasets.
    """
    stem     = Path(file_path).stem
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    image_id  = f"generic_{path_hash}_{stem}"
    return image_id, stem, "unknown", None


# ---- Dispatcher --------------------------------------------

_PARSERS = {
    "eyepacs":  _parse_eyepacs,
    "aptos":    _parse_aptos,
    "idrid":    _parse_idrid,
    "messidor2": _parse_messidor2,
    "rfmid":    _parse_rfmid,
    "odir":     _parse_odir,
    "stare":    _parse_stare,
    "drive":    _parse_drive,
}


def parse_filename_metadata(
    file_path: str,
    source_dataset_name: str,
) -> tuple[str, str, str, Optional[str]]:
    """
    Extract (image_id, patient_id, eye_laterality, disease_label)
    from a file path using the correct parser for the named dataset.

    params:
        file_path           — absolute or relative path to the image file
        source_dataset_name — one of the keys in _PARSERS above
    returns: (image_id, patient_id, laterality, disease_label)
    side effects: none — pure parsing function

    NOTE: Different datasets encode metadata very differently.
    This dispatcher routes to the right parser so the database gets
    consistent column values regardless of the source.
    """
    parser = _PARSERS.get(source_dataset_name.lower(), _parse_generic)
    return parser(file_path)
