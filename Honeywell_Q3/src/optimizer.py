"""
Optimizer Engine Module for Honeywell Autonomous Production Choke Controller.

Performs deterministic candidate evaluation, physical safety constraint checks against
operating pressure envelopes (WHP, FLP, BHP) and maximum actuation rate limits (±5%),
and calculates industrial weighted multi-objective scores.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.config import OPERATIONAL_CONSTRAINTS, OPTIMIZER_CONFIG
from src.utils import logger

class ChokeOptimizer:
    """
    Deterministic optimization engine evaluating candidate choke positions.
    """

    def __init__(self, constraints: Dict[str, float] = None, optimizer_config: Dict[str, float] = None):
        """
        Initialize Choke Optimizer.

        Args:
            constraints: Operational pressure and choke boundaries dictionary.
            optimizer_config: Multi-objective penalty weight dictionary.
        """
        self.constraints = constraints or OPERATIONAL_CONSTRAINTS
        self.config = optimizer_config or OPTIMIZER_CONFIG

    def check_constraints(
        self,
        candidate_choke: float,
        current_choke: float,
        predicted_whp: float,
        predicted_flp: float,
        predicted_bhp: float,
        predicted_oil_rate: float = 100.0
    ) -> Tuple[bool, List[str]]:
        """
        Evaluates physical safety constraints and operating limits for a candidate choke action.

        Args:
            candidate_choke: Candidate choke opening percentage (%).
            current_choke: Current choke opening percentage (%).
            predicted_whp: Predicted Wellhead Pressure (psi).
            predicted_flp: Predicted Flowline Pressure (psi).
            predicted_bhp: Predicted Bottom Hole Pressure (psi).
            predicted_oil_rate: Predicted Oil Production Rate (bbl/hr).

        Returns:
            Tuple containing boolean (is_safe) and list of constraint violation messages.
        """
        violations = []

        # 1. Choke Absolute Range Boundary Check [0%, 100%]
        if candidate_choke < self.constraints["CHOKE_MIN"] or candidate_choke > self.constraints["CHOKE_MAX"]:
            violations.append(f"Choke out of bounds [{self.constraints['CHOKE_MIN']}%, {self.constraints['CHOKE_MAX']}%]")

        # 2. Maximum Choke Movement Rate Limit Check (±5% per control interval)
        delta_choke = abs(candidate_choke - current_choke)
        if delta_choke > (self.constraints["MAX_DELTA_CHOKE"] + 1e-5):
            violations.append(f"Choke shift {delta_choke:.2f}% > max step limit ±{self.constraints['MAX_DELTA_CHOKE']}%")

        # 3. Wellhead Pressure Envelope Check (WHP_MIN <= WHP <= WHP_MAX)
        if predicted_whp < self.constraints["WHP_MIN"]:
            violations.append(f"Predicted WHP {predicted_whp:.1f} psi < MIN limit ({self.constraints['WHP_MIN']} psi)")
        elif predicted_whp > self.constraints["WHP_MAX"]:
            violations.append(f"Predicted WHP {predicted_whp:.1f} psi > MAX limit ({self.constraints['WHP_MAX']} psi)")

        # 4. Flowline Pressure Envelope Check (FLP_MIN <= FLP <= FLP_MAX)
        if predicted_flp < self.constraints["FLP_MIN"]:
            violations.append(f"Predicted FLP {predicted_flp:.1f} psi < MIN limit ({self.constraints['FLP_MIN']} psi)")
        elif predicted_flp > self.constraints["FLP_MAX"]:
            violations.append(f"Predicted FLP {predicted_flp:.1f} psi > MAX limit ({self.constraints['FLP_MAX']} psi)")

        # 5. Bottom Hole Pressure Envelope Check (BHP_MIN <= BHP <= BHP_MAX)
        if predicted_bhp < self.constraints["BHP_MIN"]:
            violations.append(f"Predicted BHP {predicted_bhp:.1f} psi < MIN limit ({self.constraints['BHP_MIN']} psi)")
        elif predicted_bhp > self.constraints["BHP_MAX"]:
            violations.append(f"Predicted BHP {predicted_bhp:.1f} psi > MAX limit ({self.constraints['BHP_MAX']} psi)")

        # 6. Physical Non-Negative Production & Pressure Stability Check
        if predicted_oil_rate < 0.0:
            violations.append(f"Unphysical negative oil production rate ({predicted_oil_rate:.1f} bbl/hr)")

        if (predicted_bhp - predicted_whp) < 10.0:
            violations.append("Insufficient pressure drawdown margin (BHP - WHP < 10 psi)")

        is_safe = len(violations) == 0
        return is_safe, violations

    def calculate_objective_score(
        self,
        predicted_oil_rate: float,
        current_oil_rate: float,
        candidate_choke: float,
        current_choke: float,
        prev_choke: float,
        predicted_whp: float,
        predicted_flp: float,
        predicted_bhp: float,
        is_safe: bool,
        num_violations: int
    ) -> float:
        """
        Calculates configurable industrial multi-objective utility score:
        Score = w_prod * Oil_Gain + w_eff * Efficiency - w_move * |ΔChoke| - w_pres * Pressure_Penalty - w_osc * Oscillation - Violations

        Returns:
            float: Multi-objective utility score.
        """
        w_prod = self.config.get("WEIGHT_OIL_GAIN", 0.60)
        w_eff = self.config.get("WEIGHT_EFFICIENCY", 0.15)
        w_move = self.config.get("WEIGHT_MOVEMENT", 0.10)
        w_pres = self.config.get("WEIGHT_PRESSURE_STABILITY", 0.10)
        w_osc = self.config.get("WEIGHT_OSCILLATION", 0.05)
        violation_penalty = self.config.get("PRESSURE_VIOLATION_PENALTY", 1000.0)

        # 1. Expected Oil Gain (relative to current state)
        oil_gain = predicted_oil_rate - current_oil_rate
        prod_score = oil_gain * w_prod * 8.0

        # 2. Production Efficiency (bbl/hr per psi drawdown)
        drawdown = max(predicted_bhp - predicted_whp, 1e-4)
        efficiency = (predicted_oil_rate / drawdown) * w_eff * 25.0

        # 3. Actuation Shift Penalty
        delta_u = abs(candidate_choke - current_choke)
        movement_penalty = delta_u * w_move * 1.5

        # 4. Pressure Margin Safety Penalty (proximity to WHP_MIN, WHP_MAX, FLP_MAX, or BHP_MIN)
        whp_min_margin = predicted_whp - self.constraints["WHP_MIN"]
        whp_max_margin = self.constraints["WHP_MAX"] - predicted_whp
        flp_max_margin = self.constraints["FLP_MAX"] - predicted_flp
        bhp_min_margin = predicted_bhp - self.constraints["BHP_MIN"]

        pressure_penalty = 0.0
        if whp_min_margin < 60.0:
            pressure_penalty += (max(0.0, 60.0 - whp_min_margin) ** 2) * w_pres * 0.8
        if whp_max_margin < 60.0:
            pressure_penalty += (max(0.0, 60.0 - whp_max_margin) ** 2) * w_pres * 0.8
        if flp_max_margin < 20.0:
            pressure_penalty += (max(0.0, 20.0 - flp_max_margin) ** 2) * w_pres * 8.0
        if bhp_min_margin < 100.0:
            pressure_penalty += (max(0.0, 100.0 - bhp_min_margin) ** 2) * w_pres * 0.3





        # 5. Oscillation Penalty (direction reversal from previous step)
        prev_delta = current_choke - prev_choke
        curr_delta = candidate_choke - current_choke
        oscillation_penalty = 0.0
        if prev_delta * curr_delta < 0:
            oscillation_penalty = abs(curr_delta) * w_osc * 5.0

        # 6. Safety Violation Penalty
        safety_penalty = 0.0
        if not is_safe:
            safety_penalty = num_violations * violation_penalty

        total_score = prod_score + efficiency - movement_penalty - pressure_penalty - oscillation_penalty - safety_penalty
        return float(total_score)


    def evaluate_candidates(
        self,
        candidate_chokes: np.ndarray,
        current_state: Dict[str, float],
        prev_choke: float,
        predicted_oil_rates: np.ndarray,
        predicted_whps: np.ndarray,
        predicted_flps: np.ndarray,
        predicted_bhps: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluates an array of candidate choke positions and scores each candidate.

        Returns:
            pd.DataFrame: Candidate evaluation ranking audit log.
        """
        current_choke = float(current_state["Choke_Position"])
        current_oil = float(current_state.get("Oil_Rate", 0.0))

        records = []
        for i in range(len(candidate_chokes)):
            u_cand = float(candidate_chokes[i])
            q_pred = float(predicted_oil_rates[i])
            whp_pred = float(predicted_whps[i])
            flp_pred = float(predicted_flps[i])
            bhp_pred = float(predicted_bhps[i])

            is_safe, violations = self.check_constraints(
                u_cand, current_choke, whp_pred, flp_pred, bhp_pred, q_pred
            )

            score = self.calculate_objective_score(
                q_pred, current_oil, u_cand, current_choke, prev_choke,
                whp_pred, flp_pred, bhp_pred, is_safe, len(violations)
            )


            eff = q_pred / max(bhp_pred - whp_pred, 1e-4)

            records.append({
                "Candidate_Choke": round(u_cand, 2),
                "Delta_Choke": round(u_cand - current_choke, 2),
                "Predicted_Oil_Rate": round(q_pred, 2),
                "Predicted_WHP": round(whp_pred, 1),
                "Predicted_FLP": round(flp_pred, 1),
                "Predicted_BHP": round(bhp_pred, 1),
                "Predicted_Drawdown": round(bhp_pred - whp_pred, 1),
                "Flow_Efficiency": round(eff, 4),
                "Is_Safe": is_safe,
                "Violation_Count": len(violations),
                "Violation_Details": "; ".join(violations) if violations else "NONE",
                "Objective_Score": round(score, 2)
            })

        df_candidates = pd.DataFrame(records)
        df_candidates = df_candidates.sort_values(by="Objective_Score", ascending=False).reset_index(drop=True)
        return df_candidates

