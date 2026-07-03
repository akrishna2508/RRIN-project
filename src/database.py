"""
src/database.py
===============
Creates and manages a lightweight SQLite database that tracks every image
in the project: where it lives on disk, its quality scores, and which
train/val/test split it has been assigned to.

WHY A DATABASE? (for beginners)
  We have potentially 100,000+ images across multiple datasets.
  A database lets us quickly ask questions like:
    "Give me all training images with quality score > 0.8"
  without loading every image into memory.
"""

import glob
import os
import sqlite3
from typing import Optional

import pandas as pd


# ---- Schema creation ---------------------------------------

def initialize_metadata_database(db_path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Open (or create) the SQLite database file and ensure the 'images' table exists.

    params: db_path (str) — filesystem path for the .sqlite file
    returns: connection, cursor  — use these to run all future SQL queries
    side effects: creates the file and table on disk if they don't exist yet
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id                    TEXT PRIMARY KEY,
            patient_id                  TEXT,
            eye_laterality              TEXT,
            source_dataset              TEXT,
            file_path                   TEXT,
            sharpness_score             REAL,
            illumination_uniformity_score REAL,
            reflection_coverage_score   REAL,
            fov_completeness_score      REAL,
            composite_quality_score     REAL,
            is_pseudo_clean             INTEGER DEFAULT 0,
            disease_label               TEXT,
            split_assignment            TEXT
        )
        """
    )
    connection.commit()
    return connection, cursor


# ---- Ingestion --------------------------------------------

def ingest_source_dataset_into_database(
    connection: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    dataset_root_path: str,
    source_dataset_name: str,
) -> int:
    """
    Walk a raw dataset folder, extract metadata from filenames/CSV files,
    and insert one row per image into the database.

    params:
        connection       — sqlite3 connection
        cursor           — sqlite3 cursor
        dataset_root_path — root folder of the dataset (e.g. "data/eyepacs")
        source_dataset_name — short name like "eyepacs", "aptos", etc.
    returns: number of new images inserted
    side effects: inserts rows into the 'images' table; safe to call twice
                  (duplicate image_ids are silently ignored via INSERT OR IGNORE)
    """
    from src.dataset_parsers import parse_filename_metadata

    # Find all image files (JPEG, PNG, TIFF)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    image_paths: list[str] = []
    for ext in extensions:
        image_paths.extend(glob.glob(
            os.path.join(dataset_root_path, "**", ext), recursive=True
        ))
        image_paths.extend(glob.glob(
            os.path.join(dataset_root_path, "**", ext.upper()), recursive=True
        ))

    rows_to_insert = []
    for file_path in image_paths:
        image_id, patient_id, eye_laterality, disease_label = parse_filename_metadata(
            file_path, source_dataset_name
        )
        rows_to_insert.append((
            image_id, patient_id, eye_laterality,
            source_dataset_name, file_path, disease_label
        ))

    if rows_to_insert:
        cursor.executemany(
            """
            INSERT OR IGNORE INTO images
                (image_id, patient_id, eye_laterality, source_dataset, file_path, disease_label)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        connection.commit()

    return len(rows_to_insert)


# ---- Quality score writes ----------------------------------

def update_quality_scores_in_database(
    connection: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    image_id: str,
    quality_score_dict: dict,
) -> None:
    """
    Write the computed quality scores for one image back into the database.

    params:
        quality_score_dict — dict with keys:
            sharpness_score, illumination_uniformity_score,
            reflection_coverage_score, fov_completeness_score,
            composite_quality_score
    side effects: updates the matching row and commits the transaction
    """
    cursor.execute(
        """
        UPDATE images
        SET sharpness_score=?,
            illumination_uniformity_score=?,
            reflection_coverage_score=?,
            fov_completeness_score=?,
            composite_quality_score=?
        WHERE image_id=?
        """,
        (
            quality_score_dict["sharpness_score"],
            quality_score_dict["illumination_uniformity_score"],
            quality_score_dict["reflection_coverage_score"],
            quality_score_dict["fov_completeness_score"],
            quality_score_dict["composite_quality_score"],
            image_id,
        ),
    )
    connection.commit()


# ---- Reads ------------------------------------------------

def load_metadata_as_dataframe(connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Pull the entire 'images' table into an in-memory pandas DataFrame.
    All subsequent splitting and filtering operate on this DataFrame.

    params: connection — sqlite3 connection
    returns: DataFrame with one row per image
    side effects: none (read-only query)
    """
    return pd.read_sql_query("SELECT * FROM images", connection)


def load_real_degraded_unpaired_dataframe(
    connection: sqlite3.Connection,
    split: str = "train",
) -> pd.DataFrame:
    """
    Load rows of images that are real & degraded (NOT pseudo-clean),
    restricted to the given split. These are used in Tier-3 domain adaptation.

    params: connection, split
    returns: DataFrame of unpaired real degraded images
    side effects: none
    """
    return pd.read_sql_query(
        "SELECT * FROM images WHERE is_pseudo_clean = 0 AND split_assignment = ?",
        connection,
        params=(split,),
    )


# ---- Split persistence ------------------------------------

def build_image_id_to_split_mapping(metadata_dataframe: pd.DataFrame) -> dict:
    """
    Build a dict mapping image_id → split_assignment from the DataFrame.
    Used before writing splits back into the database.
    """
    return dict(zip(
        metadata_dataframe["image_id"],
        metadata_dataframe["split_assignment"]
    ))


def write_split_assignments_to_database(
    connection: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    image_id_to_split_mapping: dict,
) -> None:
    """
    Persist the patient-grouped split assignments for every image into
    the database so every future stage reads a fixed, reproducible split.

    params:
        image_id_to_split_mapping — dict: image_id → "train"/"val"/"test"/"holdout"
    side effects: updates split_assignment column for all listed images; commits
    """
    update_tuples = [
        (split_value, image_id)
        for image_id, split_value in image_id_to_split_mapping.items()
    ]
    cursor.executemany(
        "UPDATE images SET split_assignment=? WHERE image_id=?",
        update_tuples,
    )
    connection.commit()
