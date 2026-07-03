"""
src/utils/logging_utils.py
===========================
Sets up a Python logger that writes to both the console and a log file.
"""

import logging
import os
from datetime import datetime


def setup_logger(log_dir: str, name: str = "rrin") -> logging.Logger:
    """
    Create a logger that prints messages to the terminal AND
    saves them to a timestamped file in log_dir.

    Usage:
        logger = setup_logger("logs/")
        logger.info("Training started")
        logger.warning("GPU memory is getting low")
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called twice
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # File handler (writes to log file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialised. Log file: {log_file}")
    return logger


def set_global_random_seeds(seed: int) -> None:
    """
    Fix all random seeds for full reproducibility.
    Call this ONCE at the very start of main().
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
