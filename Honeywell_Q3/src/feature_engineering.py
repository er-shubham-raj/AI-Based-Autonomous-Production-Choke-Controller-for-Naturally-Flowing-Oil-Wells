"""
Feature Engineering Module for Honeywell Autonomous Production Choke Controller.

Engineers 35+ industrial process features derived from well sensor telemetry:
1. Pressure Differentials & Ratios (WHP-FLP, BHP-WHP, BHP-FLP, ratios, % drops)
2. Non-linear Choke Features (Normalized choke, Choke², Choke³, lags, velocity)
3. Oil Dynamics & Momentum (Rolling mean/std, momentum, acceleration, % changes)
4. Pressure Dynamics (Rolling stats for WHP/FLP/BHP, rate of change)
5. Physical Interaction Terms (Choke x WHP, Choke x FLP, Choke x BHP, Drawdown x Choke)
6. Flow Efficiency Indicators (Oil/Drawdown, Oil/WHP, Oil/BHP, Oil/Choke)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

from src.config import FEATURE_COLUMNS
from src.utils import logger

class ProcessFeatureEngineer:
    """
    Computes 35+ industrial process features for oil well dynamics modelling.
    """

    def __init__(self, rolling_window: int = 3, eps: float = 1e-4):
        """
        Initialize Feature Engineer.

        Args:
            rolling_window: Window size for rolling statistics.
            eps: Epsilon value to prevent divide-by-zero errors.
        """
        self.rolling_window = rolling_window
        self.eps = eps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers 35+ process features on input telemetry DataFrame.

        Args:
            df: Input DataFrame containing raw/preprocessed telemetry channels.

        Returns:
            pd.DataFrame: DataFrame augmented with engineered features.
        """
        logger.info(f"Engineering 35+ process analytics features (rolling window = {self.rolling_window})...")
        df_feat = df.copy()

        # 1. Base Sensor Readings validation
        for col in ["Choke_Position", "Wellhead_Pressure", "Flowline_Pressure", "Bottom_Hole_Pressure", "Oil_Rate"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        whp = df_feat["Wellhead_Pressure"]
        flp = df_feat["Flowline_Pressure"]
        bhp = df_feat["Bottom_Hole_Pressure"]
        choke = df_feat["Choke_Position"]
        oil = df_feat["Oil_Rate"]

        # 2. Pressure Differentials & Ratios
        df_feat["Pressure_Diff_WHP_FLP"] = whp - flp
        df_feat["Pressure_Diff_BHP_WHP"] = bhp - whp
        df_feat["Pressure_Diff_BHP_FLP"] = bhp - flp
        
        df_feat["Pressure_Ratio_WHP_FLP"] = whp / (flp + self.eps)
        df_feat["Pressure_Ratio_BHP_WHP"] = bhp / (whp + self.eps)
        df_feat["Pressure_Ratio_BHP_FLP"] = bhp / (flp + self.eps)
        df_feat["Pressure_Drop_Percent"] = (whp - flp) / (whp + self.eps) * 100.0

        # 3. Choke Non-Linear & Derivative Features
        df_feat["Normalized_Choke"] = choke / 100.0
        df_feat["Choke_Squared"] = (choke / 100.0) ** 2
        df_feat["Choke_Cubed"] = (choke / 100.0) ** 3
        df_feat["Lag_Choke_Position"] = choke.shift(1).fillna(choke)
        df_feat["Choke_Change"] = choke - df_feat["Lag_Choke_Position"]
        df_feat["Choke_Velocity"] = df_feat["Choke_Change"].rolling(window=self.rolling_window, min_periods=1).mean()

        # 4. Oil Telemetry & Momentum Features
        df_feat["Oil_Rolling_Mean"] = oil.rolling(window=self.rolling_window, min_periods=1).mean()
        df_feat["Oil_Rolling_Std"] = oil.rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0)
        
        lag_oil = oil.shift(1).fillna(oil)
        df_feat["Oil_Momentum"] = oil - lag_oil
        df_feat["Oil_Acceleration"] = df_feat["Oil_Momentum"] - df_feat["Oil_Momentum"].shift(1).fillna(0.0)
        df_feat["Oil_Percent_Change"] = (oil - lag_oil) / (lag_oil + self.eps) * 100.0

        # 5. Pressure Dynamics & Rolling Window Statistics
        df_feat["Rolling_Mean_WHP"] = whp.rolling(window=self.rolling_window, min_periods=1).mean()
        df_feat["Rolling_Mean_FLP"] = flp.rolling(window=self.rolling_window, min_periods=1).mean()
        df_feat["Rolling_Mean_BHP"] = bhp.rolling(window=self.rolling_window, min_periods=1).mean()

        df_feat["Rolling_Std_WHP"] = whp.rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0)
        df_feat["Rolling_Std_FLP"] = flp.rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0)
        df_feat["Rolling_Std_BHP"] = bhp.rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0)

        df_feat["Rate_Change_WHP"] = whp - whp.shift(1).fillna(whp)
        df_feat["Rate_Change_FLP"] = flp - flp.shift(1).fillna(flp)
        df_feat["Rate_Change_BHP"] = bhp - bhp.shift(1).fillna(bhp)

        # 6. Physical Interaction Terms
        df_feat["Choke_x_WHP"] = (choke / 100.0) * whp
        df_feat["Choke_x_FLP"] = (choke / 100.0) * flp
        df_feat["Choke_x_BHP"] = (choke / 100.0) * bhp
        
        drawdown = bhp - whp
        df_feat["Drawdown_x_Choke"] = drawdown * (choke / 100.0)
        df_feat["Pressure_Ratio_x_Choke"] = df_feat["Pressure_Ratio_WHP_FLP"] * (choke / 100.0)

        # 7. Flow Efficiency Features
        df_feat["Oil_per_Drawdown"] = oil / (drawdown + self.eps)
        df_feat["Oil_per_WHP"] = oil / (whp + self.eps)
        df_feat["Oil_per_BHP"] = oil / (bhp + self.eps)
        df_feat["Flow_Efficiency"] = oil / (choke + self.eps)

        # Handle initial NaN values cleanly
        df_feat = df_feat.bfill().ffill().fillna(0.0)

        logger.info(f"Successfully engineered {len(FEATURE_COLUMNS)} features for dataset shape {df_feat.shape}.")
        return df_feat

    def transform_single_candidate(
        self,
        current_state: Dict[str, float],
        cand_choke: float,
        cand_whp: float,
        cand_flp: float,
        cand_bhp: float
    ) -> Dict[str, float]:
        """
        Constructs single candidate feature dictionary for real-time MPC candidate evaluation.
        Calculates all 35+ features dynamically for given candidate choke and candidate pressures.
        """
        eps = self.eps
        curr_choke = float(current_state.get("Choke_Position", cand_choke))
        curr_oil = float(current_state.get("Oil_Rate", 0.0))

        diff_whp_flp = cand_whp - cand_flp
        diff_bhp_whp = cand_bhp - cand_whp
        diff_bhp_flp = cand_bhp - cand_flp

        ratio_whp_flp = cand_whp / (cand_flp + eps)
        ratio_bhp_whp = cand_bhp / (cand_whp + eps)
        ratio_bhp_flp = cand_bhp / (cand_flp + eps)
        drop_pct = (cand_whp - cand_flp) / (cand_whp + eps) * 100.0

        norm_choke = cand_choke / 100.0
        choke_sq = norm_choke ** 2
        choke_cu = norm_choke ** 3
        choke_change = cand_choke - curr_choke
        choke_vel = choke_change

        oil_momentum = 0.0
        oil_accel = 0.0
        oil_pct_change = 0.0

        roll_whp = float(current_state.get("Rolling_Mean_WHP", cand_whp))
        roll_flp = float(current_state.get("Rolling_Mean_FLP", cand_flp))
        roll_bhp = float(current_state.get("Rolling_Mean_BHP", cand_bhp))

        std_whp = float(current_state.get("Rolling_Std_WHP", 0.0))
        std_flp = float(current_state.get("Rolling_Std_FLP", 0.0))
        std_bhp = float(current_state.get("Rolling_Std_BHP", 0.0))

        rate_whp = cand_whp - float(current_state.get("Wellhead_Pressure", cand_whp))
        rate_flp = cand_flp - float(current_state.get("Flowline_Pressure", cand_flp))
        rate_bhp = cand_bhp - float(current_state.get("Bottom_Hole_Pressure", cand_bhp))

        choke_whp = norm_choke * cand_whp
        choke_flp = norm_choke * cand_flp
        choke_bhp = norm_choke * cand_bhp
        drawdown_choke = diff_bhp_whp * norm_choke
        ratio_choke = ratio_whp_flp * norm_choke

        oil_drawdown = curr_oil / (diff_bhp_whp + eps)
        oil_whp = curr_oil / (cand_whp + eps)
        oil_bhp = curr_oil / (cand_bhp + eps)
        flow_eff = curr_oil / (cand_choke + eps)

        return {
            "Choke_Position": cand_choke,
            "Wellhead_Pressure": cand_whp,
            "Flowline_Pressure": cand_flp,
            "Bottom_Hole_Pressure": cand_bhp,
            "Pressure_Diff_WHP_FLP": diff_whp_flp,
            "Pressure_Diff_BHP_WHP": diff_bhp_whp,
            "Pressure_Diff_BHP_FLP": diff_bhp_flp,
            "Pressure_Ratio_WHP_FLP": ratio_whp_flp,
            "Pressure_Ratio_BHP_WHP": ratio_bhp_whp,
            "Pressure_Ratio_BHP_FLP": ratio_bhp_flp,
            "Pressure_Drop_Percent": drop_pct,
            "Normalized_Choke": norm_choke,
            "Choke_Squared": choke_sq,
            "Choke_Cubed": choke_cu,
            "Lag_Choke_Position": curr_choke,
            "Choke_Change": choke_change,
            "Choke_Velocity": choke_vel,
            "Oil_Rolling_Mean": curr_oil,
            "Oil_Rolling_Std": 0.0,
            "Oil_Momentum": oil_momentum,
            "Oil_Acceleration": oil_accel,
            "Oil_Percent_Change": oil_pct_change,
            "Rolling_Mean_WHP": roll_whp,
            "Rolling_Mean_FLP": roll_flp,
            "Rolling_Mean_BHP": roll_bhp,
            "Rolling_Std_WHP": std_whp,
            "Rolling_Std_FLP": std_flp,
            "Rolling_Std_BHP": std_bhp,
            "Rate_Change_WHP": rate_whp,
            "Rate_Change_FLP": rate_flp,
            "Rate_Change_BHP": rate_bhp,
            "Choke_x_WHP": choke_whp,
            "Choke_x_FLP": choke_flp,
            "Choke_x_BHP": choke_bhp,
            "Drawdown_x_Choke": drawdown_choke,
            "Pressure_Ratio_x_Choke": ratio_choke,
            "Oil_per_Drawdown": oil_drawdown,
            "Oil_per_WHP": oil_whp,
            "Oil_per_BHP": oil_bhp,
            "Flow_Efficiency": flow_eff
        }

    def get_feature_names(self) -> List[str]:
        """
        Returns list of engineered feature names used for model inputs.
        """
        return FEATURE_COLUMNS

def engineer_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """
    Convenience wrapper function for ProcessFeatureEngineer.
    """
    engineer = ProcessFeatureEngineer(rolling_window=rolling_window)
    return engineer.transform(df)

