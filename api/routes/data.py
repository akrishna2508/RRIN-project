"""
api/routes/data.py
==================
API endpoints for the data preparation pipeline.

ENDPOINTS:
  POST /api/v1/data/ingest        — Register a dataset folder in the database
  POST /api/v1/data/quality-score — Compute quality scores for all images
  POST /api/v1/data/create-splits — Assign patient-grouped train/val/test splits
  GET  /api/v1/data/summary       — Show counts by split and dataset

HOW TO USE (for beginners):
  You call these endpoints IN ORDER before training:
    1. /ingest       (once per dataset)
    2. /quality-score
    3. /create-splits
  Then start training via /api/v1/training/start
"""

import sqlite3
from fastapi import APIRouter, HTTPException

from api.schemas import (
    IngestRequest, IngestResponse,
    QualityScoringRequest, QualityScoringResponse,
    SplitRequest, SplitResponse,
)
from src.config import METADATA_DB_PATH
from src.database import (
    initialize_metadata_database,
    ingest_source_dataset_into_database,
    load_metadata_as_dataframe,
    write_split_assignments_to_database,
    build_image_id_to_split_mapping,
)
from src.quality_scoring import compute_and_attach_all_quality_scores, select_pseudo_clean_pool
from src.splits import assign_patient_grouped_splits, verify_no_patient_overlap_across_splits

router = APIRouter(prefix="/data", tags=["Data Pipeline"])


def _get_db():
    """Open a fresh DB connection (one per request, to avoid threading issues)."""
    return initialize_metadata_database(METADATA_DB_PATH)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_dataset(request: IngestRequest):
    """
    Register a dataset folder in the metadata database.

    Call this once for EACH dataset you have downloaded.
    The function walks the folder, finds all image files, and records
    them in the database with their patient ID and eye laterality.

    Example call:
        POST /api/v1/data/ingest
        Body: {"dataset_name": "eyepacs", "dataset_path": "/data/eyepacs"}
    """
    import os
    if not os.path.isdir(request.dataset_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {request.dataset_path}\n"
                   f"Please make sure the dataset is downloaded to this path."
        )

    connection, cursor = _get_db()
    try:
        n_ingested = ingest_source_dataset_into_database(
            connection, cursor,
            request.dataset_path,
            request.dataset_name.value
        )
        connection.close()
        return IngestResponse(
            dataset_name=request.dataset_name.value,
            images_ingested=n_ingested,
            message=f"Successfully registered {n_ingested} images from '{request.dataset_name.value}'."
        )
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-score", response_model=QualityScoringResponse)
async def compute_quality_scores(request: QualityScoringRequest):
    """
    Compute image quality scores for all images in the database.

    This can take a long time (minutes to hours depending on dataset size).
    Results are cached in the database so this only needs to run ONCE.

    Quality scores measure:
      - Sharpness (blur detection)
      - Illumination uniformity (vignetting detection)
      - Reflection coverage (specular highlight detection)
      - Field-of-view completeness (cropping detection)

    Only images in the TOP 25% by composite score become training targets.
    """
    connection, cursor = _get_db()
    try:
        df = load_metadata_as_dataframe(connection)

        if not request.recompute_all:
            # Only score images that don't have scores yet
            unscored = df[df["composite_quality_score"].isna()]
            print(f"  Scoring {len(unscored)} unscored images (skipping {len(df) - len(unscored)} already scored).")
            df = unscored

        if df.empty:
            connection.close()
            return QualityScoringResponse(
                images_scored=0,
                pseudo_clean_count=0,
                message="All images already scored. Set recompute_all=true to re-score."
            )

        df = compute_and_attach_all_quality_scores(df, connection, cursor)
        df = select_pseudo_clean_pool(df)

        # Write is_pseudo_clean flag back to DB
        for _, row in df.iterrows():
            cursor.execute(
                "UPDATE images SET is_pseudo_clean=? WHERE image_id=?",
                (int(row.get("is_pseudo_clean", False)), row["image_id"])
            )
        connection.commit()
        pseudo_count = int(df["is_pseudo_clean"].sum()) if "is_pseudo_clean" in df.columns else 0

        connection.close()
        return QualityScoringResponse(
            images_scored=len(df),
            pseudo_clean_count=pseudo_count,
            message=f"Scored {len(df)} images. {pseudo_count} selected as pseudo-clean targets."
        )
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-splits", response_model=SplitResponse)
async def create_splits(request: SplitRequest):
    """
    Assign patient-grouped train / val / test splits.

    This must run AFTER quality scoring.

    IMPORTANT: Splits are assigned at the PATIENT level, not the image level.
    All images from the same patient always go to the same split.
    This prevents data leakage (the AI memorising patient anatomy).

    MESSIDOR-2 images are always assigned to 'holdout' (never seen during training).
    """
    connection, cursor = _get_db()
    try:
        df = load_metadata_as_dataframe(connection)

        if "is_pseudo_clean" not in df.columns or df["is_pseudo_clean"].isna().all():
            connection.close()
            raise HTTPException(
                status_code=400,
                detail="Quality scores not yet computed. Call /data/quality-score first."
            )

        df = assign_patient_grouped_splits(
            df,
            train_fraction=request.train_fraction,
            val_fraction=request.val_fraction,
            random_seed=request.random_seed,
        )
        is_leakage_free = verify_no_patient_overlap_across_splits(df)

        mapping = build_image_id_to_split_mapping(df)
        write_split_assignments_to_database(connection, cursor, mapping)

        split_counts = df["split_assignment"].value_counts().to_dict()
        connection.close()

        return SplitResponse(
            split_counts=split_counts,
            leakage_check="PASSED — no patient overlap across splits" if is_leakage_free else "WARNING",
            message="Splits created successfully. You can now start training."
        )
    except HTTPException:
        connection.close()
        raise
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_data_summary():
    """Get a summary of the current database contents."""
    connection, _ = _get_db()
    try:
        df = load_metadata_as_dataframe(connection)
        connection.close()
        summary = {
            "total_images":      len(df),
            "by_dataset":        df["source_dataset"].value_counts().to_dict() if "source_dataset" in df.columns else {},
            "by_split":          df["split_assignment"].value_counts().to_dict() if "split_assignment" in df.columns else {},
            "pseudo_clean_count": int(df["is_pseudo_clean"].sum()) if "is_pseudo_clean" in df.columns else 0,
            "has_quality_scores": bool((~df["composite_quality_score"].isna()).any()) if "composite_quality_score" in df.columns else False,
        }
        return summary
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=str(e))
