"""
Preprocessing Module for Honeywell Autonomous Production Choke Controller.

Performs data cleaning, missing value handling (forward-fill / median),
sensor noise clipping, and physical sanity bounds filtering on telemetry variables.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

from src.config import OPERATIONAL_CONSTRAINTS, PROCESSED_DATA_DIR
from src.utils import logger, ensure_directories_exist

class DataPreprocessor:
    """
    Handles data cleaning, signal smoothing, missing value imputation,
    and physical pressure envelope bounds filtering.
    """

    def __init__(self, clip_outliers: bool = True, smooth_signals: bool = False):
        """
        Initialize Preprocessor.

        Args:
            clip_outliers: Whether to clip sensor values to physical limits.
            smooth_signals: Whether to apply exponential moving average filter.
        """
        self.clip_outliers = clip_outliers
        self.smooth_signals = smooth_signals
        ensure_directories_exist()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs full preprocessing workflow on telemetry DataFrame.

        Args:
            df: Input raw DataFrame.

        Returns:
            pd.DataFrame: Cleaned & preprocessed DataFrame.
        """
        logger.info(f"Starting data preprocessing on {len(df)} telemetry rows...")
        df_clean = df.copy()

        # 1. Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # 2. Enforce physical range limits
        if self.clip_outliers:
            df_clean = self._apply_physical_bounds(df_clean)

        # 3. Apply optional signal smoothing for high-frequency noise reduction
        if self.smooth_signals:
            df_clean = self._apply_signal_smoothing(df_clean)

        # Save processed baseline
        output_path = PROCESSED_DATA_DIR / "processed_well_telemetry.csv"
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Preprocessing completed. Cleaned dataset saved to: {output_path}")

        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using forward-fill followed by backward-fill or median.
        """
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Detected {missing_count} missing values. Applying forward/backward fill...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            
            # Fill remaining with median if any
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
        else:
            logger.info("No missing values detected in sensor telemetry channels.")
        return df

    def _apply_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clips sensor channels to realistic physical operating boundaries.
        """
        df["Choke_Position"] = np.clip(
            df["Choke_Position"],
            OPERATIONAL_CONSTRAINTS["CHOKE_MIN"],
            OPERATIONAL_CONSTRAINTS["CHOKE_MAX"]
        )
        df["Oil_Rate"] = np.maximum(df["Oil_Rate"], 0.0)
        df["Wellhead_Pressure"] = np.clip(df["Wellhead_Pressure"], 0.0, 5000.0)
        df["Flowline_Pressure"] = np.clip(df["Flowline_Pressure"], 0.0, 2000.0)
        df["Bottom_Hole_Pressure"] = np.clip(df["Bottom_Hole_Pressure"], 0.0, 10000.0)

        logger.info("Physical sensor boundaries enforced.")
        return df

    def _apply_signal_smoothing(self, df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
        """
        Applies Exponential Moving Average (EMA) to reduce high-frequency pressure oscillations.
        """
        sensor_cols = ["Wellhead_Pressure", "Flowline_Pressure", "Bottom_Hole_Pressure", "Oil_Rate"]
        for col in sensor_cols:
            if col in df.columns:
                df[col] = df[col].ewm(alpha=alpha, adjust=False).mean()
        logger.info("Signal smoothing applied across pressure and production channels.")
        return df

def preprocess_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper for DataPreprocessor.
    """
    preprocessor = DataPreprocessor()
    return preprocessor.fit_transform(df)
