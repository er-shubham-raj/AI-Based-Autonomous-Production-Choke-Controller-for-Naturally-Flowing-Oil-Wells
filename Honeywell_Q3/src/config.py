"""
Configuration Module for Honeywell Autonomous Production Choke Controller.

Contains lightweight configuration dictionaries for well operational envelope,
ML hyperparameters, 35+ industrial process features, and multi-objective optimizer settings.
"""

from typing import Dict, Any, List
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
REPORT_DIR = BASE_DIR / "report"

# Data schema mapping - expected columns from historical dataset
DATA_COLUMNS: Dict[str, str] = {
    "TIMESTAMP": "Time",
    "CHOKE": "Choke_Position",          # Choke Opening % [0 - 100]
    "OIL_RATE": "Oil_Rate",             # Production Rate (bbl/hr)
    "WHP": "Wellhead_Pressure",         # Wellhead Pressure (psi)
    "FLP": "Flowline_Pressure",         # Flowline Pressure (psi)
    "BHP": "Bottom_Hole_Pressure"       # Bottom Hole Pressure (psi)
}

# Operational Pressure and Actuation Limits (Honeywell Well Safety Envelope)
OPERATIONAL_CONSTRAINTS: Dict[str, float] = {
    # Choke Limits
    "CHOKE_MIN": 0.0,                   # Minimum allowable choke %
    "CHOKE_MAX": 100.0,                 # Maximum allowable choke %
    "MAX_DELTA_CHOKE": 5.0,             # Maximum allowed choke shift per control interval (±5%)
    "DEFAULT_CHOKE_STEP": 0.5,          # Discretization step size for candidate generation (%)

    # Pressure Safety Envelope (psi) - Configurable per well profile
    "WHP_MIN": 250.0,                   # Minimum safe Wellhead Pressure to avoid liquid loading/choking
    "WHP_MAX": 1200.0,                  # Maximum safe Wellhead Pressure (piping pressure rating limit)
    "FLP_MIN": 50.0,                    # Minimum Flowline Pressure
    "FLP_MAX": 450.0,                   # Maximum Flowline Pressure (separator inlet limit)
    "BHP_MIN": 800.0,                   # Minimum Bottom Hole Pressure (bubble point / reservoir drawdown safety limit)
    "BHP_MAX": 3500.0                   # Maximum Bottom Hole Pressure
}

# Optimization Engine Penalties and Weights (Industrial Multi-Objective Utility Formulation)
OPTIMIZER_CONFIG: Dict[str, float] = {
    "WEIGHT_OIL_GAIN": 0.60,            # Weight for expected oil production gain
    "WEIGHT_EFFICIENCY": 0.15,          # Weight for production efficiency (bbl/hr per psi drawdown)
    "WEIGHT_MOVEMENT": 0.10,            # Penalty weight for choke movement amplitude
    "WEIGHT_PRESSURE_STABILITY": 0.10,   # Weight for maintaining safe margin from pressure boundaries
    "WEIGHT_OSCILLATION": 0.05,         # Penalty weight for direction changes (choke hunting)
    "PRESSURE_VIOLATION_PENALTY": 1000.0 # Extreme penalty score for candidates violating safety bounds
}

# Machine Learning Hyperparameters & Training Settings (4 Regressors Benchmark)
MODEL_CONFIG: Dict[str, Any] = {
    "RANDOM_SEED": 42,
    "TEST_SIZE": 0.2,
    "CV_FOLDS": 5,
    "TARGET_VARIABLE": "Oil_Rate",
    "RF_N_ESTIMATORS": 150,
    "RF_MAX_DEPTH": 12,
    "ET_N_ESTIMATORS": 150,
    "ET_MAX_DEPTH": 12,
    "GB_N_ESTIMATORS": 150,
    "GB_MAX_DEPTH": 6,
    "GB_LEARNING_RATE": 0.05
}

# 35+ Industrial Process Feature Definitions
FEATURE_COLUMNS: List[str] = [
    # 1. Base Sensor Readings
    "Choke_Position",
    "Wellhead_Pressure",
    "Flowline_Pressure",
    "Bottom_Hole_Pressure",
    
    # 2. Pressure Differentials & Ratios
    "Pressure_Diff_WHP_FLP",
    "Pressure_Diff_BHP_WHP",
    "Pressure_Diff_BHP_FLP",
    "Pressure_Ratio_WHP_FLP",
    "Pressure_Ratio_BHP_WHP",
    "Pressure_Ratio_BHP_FLP",
    "Pressure_Drop_Percent",
    
    # 3. Choke Non-Linear & Derivative Features
    "Normalized_Choke",
    "Choke_Squared",
    "Choke_Cubed",
    "Lag_Choke_Position",
    "Choke_Change",
    "Choke_Velocity",
    
    # 4. Oil Telemetry & Momentum Features
    "Oil_Rolling_Mean",
    "Oil_Rolling_Std",
    "Oil_Momentum",
    "Oil_Acceleration",
    "Oil_Percent_Change",
    
    # 5. Pressure Dynamics & Rolling Window Statistics
    "Rolling_Mean_WHP",
    "Rolling_Mean_FLP",
    "Rolling_Mean_BHP",
    "Rolling_Std_WHP",
    "Rolling_Std_FLP",
    "Rolling_Std_BHP",
    "Rate_Change_WHP",
    "Rate_Change_FLP",
    "Rate_Change_BHP",
    
    # 6. Physical Interaction Terms
    "Choke_x_WHP",
    "Choke_x_FLP",
    "Choke_x_BHP",
    "Drawdown_x_Choke",
    "Pressure_Ratio_x_Choke",
    
    # 7. Flow Efficiency Features
    "Oil_per_Drawdown",
    "Oil_per_WHP",
    "Oil_per_BHP",
    "Flow_Efficiency"
]

def get_config() -> Dict[str, Any]:
    """
    Returns unified configuration dictionary.
    """
    return {
        "paths": {
            "base": BASE_DIR,
            "raw_data": RAW_DATA_DIR,
            "processed_data": PROCESSED_DATA_DIR,
            "models": MODELS_DIR,
            "plots": PLOTS_DIR
        },
        "columns": DATA_COLUMNS,
        "constraints": OPERATIONAL_CONSTRAINTS,
        "optimizer": OPTIMIZER_CONFIG,
        "model": MODEL_CONFIG,
        "features": FEATURE_COLUMNS
    }

