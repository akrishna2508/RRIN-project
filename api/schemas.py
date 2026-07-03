"""
api/schemas.py
==============
Pydantic data models for the FastAPI request and response bodies.

WHAT IS THIS? (for beginners)
  These are "blueprints" for the data that goes IN and OUT of the API.
  FastAPI uses them to automatically:
    - Validate that requests contain the required fields
    - Show you what the API expects in its documentation (/docs)
    - Convert Python objects to JSON automatically
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---- Enums ------------------------------------------------

class SplitName(str, Enum):
    train  = "train"
    val    = "val"
    test   = "test"
    holdout = "holdout"


class DatasetName(str, Enum):
    eyepacs  = "eyepacs"
    aptos    = "aptos"
    idrid    = "idrid"
    messidor2 = "messidor2"
    rfmid    = "rfmid"
    odir     = "odir"
    stare    = "stare"
    drive    = "drive"


class TrainingStatus(str, Enum):
    idle       = "idle"
    running    = "running"
    paused     = "paused"
    completed  = "completed"
    failed     = "failed"


# ---- Data Pipeline Schemas --------------------------------

class IngestRequest(BaseModel):
    """Request body for ingesting a dataset into the metadata database."""
    dataset_name: DatasetName = Field(..., description="Name of the dataset to ingest")
    dataset_path: str         = Field(..., description="Absolute path to the dataset folder on disk")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "eyepacs",
                "dataset_path": "/data/eyepacs"
            }
        }


class IngestResponse(BaseModel):
    """Response after ingesting a dataset."""
    dataset_name:     str
    images_ingested:  int
    message:          str


class QualityScoringRequest(BaseModel):
    """Request to compute quality scores for all unscored images."""
    recompute_all: bool = Field(
        default=False,
        description="If True, recompute scores for images that already have scores"
    )


class QualityScoringResponse(BaseModel):
    """Response after quality scoring."""
    images_scored:       int
    pseudo_clean_count:  int
    message:             str


class SplitRequest(BaseModel):
    """Request to assign train/val/test splits."""
    train_fraction:        float = Field(default=0.85, ge=0.5, le=0.95)
    val_fraction:          float = Field(default=0.10, ge=0.02, le=0.3)
    quality_quantile:      float = Field(default=0.75, ge=0.5, le=0.99)
    random_seed:           int   = Field(default=42)

    class Config:
        json_schema_extra = {
            "example": {
                "train_fraction": 0.85,
                "val_fraction": 0.10,
                "quality_quantile": 0.75,
                "random_seed": 42
            }
        }


class SplitResponse(BaseModel):
    """Response after split assignment."""
    split_counts:  dict[str, int]
    leakage_check: str
    message:       str


# ---- Training Schemas -------------------------------------

class TrainingRequest(BaseModel):
    """Request to start or resume training."""
    num_epochs:           int   = Field(default=200, ge=1, le=1000)
    batch_size:           int   = Field(default=4,   ge=1, le=32)
    learning_rate:        float = Field(default=2e-4, gt=0)
    run_domain_adaptation: bool = Field(default=False)
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint to resume from (None = start fresh)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "num_epochs": 200,
                "batch_size": 4,
                "learning_rate": 0.0002,
                "run_domain_adaptation": False,
                "resume_from_checkpoint": None
            }
        }


class TrainingStatusResponse(BaseModel):
    """Current training status."""
    status:          TrainingStatus
    current_epoch:   Optional[int]
    total_epochs:    Optional[int]
    best_ssim:       Optional[float]
    best_psnr:       Optional[float]
    latest_losses:   Optional[dict[str, float]]
    elapsed_seconds: Optional[float]
    message:         str


class TrainingStopResponse(BaseModel):
    """Response after stopping training."""
    message:       str
    final_epoch:   Optional[int]
    best_checkpoint: Optional[str]


# ---- Inference Schemas ------------------------------------

class RestoreResponse(BaseModel):
    """Response after restoring a single image."""
    output_path:      str
    input_filename:   str
    processing_time_ms: float
    model_checkpoint: str
    uncertainty_computed: bool = False


class BatchRestoreRequest(BaseModel):
    """Request for batch restoration of a folder."""
    input_folder:         str
    output_folder:        str
    checkpoint_path:      str   = Field(default="checkpoints/best.pt")
    compute_uncertainty:  bool  = Field(default=False)
    n_mc_samples:         int   = Field(default=10, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "input_folder":   "data/to_restore",
                "output_folder":  "data/restored",
                "checkpoint_path": "checkpoints/best.pt",
                "compute_uncertainty": False,
                "n_mc_samples": 10
            }
        }


class BatchRestoreResponse(BaseModel):
    """Response after batch restoration completes."""
    images_processed:    int
    output_folder:       str
    output_paths:        list[str]
    processing_time_s:   float
    message:             str


# ---- Evaluation Schemas -----------------------------------

class EvaluationRequest(BaseModel):
    """Request to run final evaluation on the test set."""
    checkpoint_path:     str  = Field(default="checkpoints/best.pt")
    run_downstream_eval: bool = Field(default=False)


class EvaluationResponse(BaseModel):
    """Summary of test-set evaluation results."""
    mean_psnr:         Optional[float]
    std_psnr:          Optional[float]
    mean_ssim:         Optional[float]
    std_ssim:          Optional[float]
    mean_lpips:        Optional[float]
    mean_vessel_dice:  Optional[float]
    n_images_evaluated: int
    results_csv_path:  str
    message:           str


# ---- Generic API Response ---------------------------------

class HealthResponse(BaseModel):
    """Health check response."""
    status:           str
    device:           str
    checkpoint_exists: bool
    database_exists:  bool
    version:          str = "1.0.0"
