"""
Utility Module for Honeywell Autonomous Production Choke Controller.

Provides logging setup, metrics calculation (MAE, RMSE, R²), directory validation,
and deterministic seed initialization.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR, PLOTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

def setup_logger(name: str = "Honeywell_Choke_Controller", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized industrial logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

def ensure_directories_exist() -> None:
    """
    Ensures all required project directories exist on the filesystem.
    """
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory verified: {d}")

def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python and NumPy for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Deterministic seed initialized: {seed}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates regression metrics: MAE, RMSE, R² Score.

    Args:
        y_true: Array of actual target values.
        y_pred: Array of predicted target values.

    Returns:
        Dictionary containing MAE, RMSE, and R2 score.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
