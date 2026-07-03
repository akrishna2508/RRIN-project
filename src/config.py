"""
src/config.py
=============
Loads the YAML config file and exposes every constant as a Python variable.
All other modules import from here so there is a single source of truth.

HOW IT WORKS (for beginners):
  When any part of the program needs a setting (like BATCH_SIZE),
  it imports it from here. This file reads config.yaml on startup.
"""

import os
import torch
import yaml
from pathlib import Path

# ---- Load the YAML file ------------------------------------
_CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load once at import time
_C = _load_yaml(_CONFIG_PATH)

# ---- Paths -------------------------------------------------
DATASET_PATHS: dict = _C.get("dataset_paths", {})
METADATA_DB_PATH: str = _C.get("metadata_db_path", "metadata/retina_restoration_metadata.sqlite")
CHECKPOINT_DIR: str = _C.get("checkpoint_dir", "checkpoints/")
LOG_DIR: str = _C.get("log_dir", "logs/")

# Make sure output directories exist
Path(METADATA_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ---- Image settings ----------------------------------------
IMAGE_SIZE: int = _C.get("image_size", 256)
CROP_SIZE: int = _C.get("crop_size", 256)
INPUT_CHANNELS: int = _C.get("input_channels", 4)
OUTPUT_CHANNELS: int = _C.get("output_channels", 3)

# ---- Model architecture ------------------------------------
BASE_FILTERS: int = _C.get("base_filters", 64)
NUM_RESIDUAL_BLOCKS: int = _C.get("num_residual_blocks", 6)
ATTENTION_GATE_ENABLED: bool = _C.get("attention_gate_enabled", True)

# ---- Training hyper-parameters ----------------------------
BATCH_SIZE: int = _C.get("batch_size", 4)
NUM_WORKERS: int = _C.get("num_workers", 2)
NUM_EPOCHS: int = _C.get("num_epochs", 200)
LR_CONSTANT_EPOCHS: int = _C.get("lr_constant_epochs", 100)
LR_DECAY_EPOCHS: int = _C.get("lr_decay_epochs", 100)

LEARNING_RATE: float = float(_C.get("learning_rate", 2e-4))
ADAM_BETA1: float = float(_C.get("adam_beta1", 0.5))
ADAM_BETA2: float = float(_C.get("adam_beta2", 0.999))
ADAM_EPSILON: float = float(_C.get("adam_epsilon", 1e-8))
WEIGHT_DECAY: float = float(_C.get("weight_decay", 1e-5))

# ---- Loss weights ------------------------------------------
LAMBDA_ADV: float = float(_C.get("lambda_adv", 1.0))
LAMBDA_L1: float = float(_C.get("lambda_l1", 100.0))
LAMBDA_SSIM: float = float(_C.get("lambda_ssim", 10.0))
LAMBDA_PERCEPTUAL: float = float(_C.get("lambda_perceptual", 10.0))
LAMBDA_CYCLE: float = float(_C.get("lambda_cycle", 10.0))

# ---- Training stability ------------------------------------
LABEL_SMOOTHING_REAL_TARGET: float = float(_C.get("label_smoothing_real_target", 0.9))
EARLY_STOPPING_PATIENCE: int = _C.get("early_stopping_patience", 15)

# ---- VGG perceptual loss layers ----------------------------
VGG_PERCEPTUAL_LAYERS: list = _C.get("vgg_perceptual_layers", ["relu2_2", "relu3_3"])

# ---- Random seed -------------------------------------------
RANDOM_SEED: int = _C.get("random_seed", 42)

# ---- Data splits -------------------------------------------
TRAIN_FRACTION: float = float(_C.get("train_fraction", 0.85))
VAL_FRACTION: float = float(_C.get("val_fraction", 0.10))
QUALITY_QUANTILE_THRESHOLD: float = float(_C.get("quality_quantile_threshold", 0.75))

# ---- Device (GPU if available, otherwise CPU) --------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
