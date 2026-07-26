"""
Controller Module for Honeywell Autonomous Production Choke Controller.

Implements closed-loop Model Predictive Control (MPC) decision loop with dynamic process simulation:
1. Current Well Telemetry & State Intake
2. Action Space Candidate Generation (u ∈ [u_k - 5%, u_k + 5%])
3. Dynamic Closed-Loop Process Simulation (Hydrodynamic pressure response propagation)
4. Dynamic 35+ Industrial Feature Construction
5. Surrogate ML Oil Rate Prediction
6. Physical Safety Constraint Filtering (WHP, FLP, BHP Pressure Envelopes)
7. Configurable Multi-Objective Utility Score Optimization
8. Optimal Choke Recommendation with Explainable AI (XAI) Audit Log & Confidence Score
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import OPERATIONAL_CONSTRAINTS, MODELS_DIR, FEATURE_COLUMNS
from src.optimizer import ChokeOptimizer
from src.feature_engineering import ProcessFeatureEngineer
from src.utils import logger, ensure_directories_exist

class AutonomousChokeController:
    """
    Industrial Autonomous Production Choke Controller (Closed-Loop Model Predictive Control).
    """

    def __init__(self, step_size: float = 0.5):
        """
        Initialize Controller.

        Args:
            step_size: Discrete candidate step size resolution (default 0.5%).
        """
        self.step_size = step_size
        self.optimizer = ChokeOptimizer()
        self.feature_engineer = ProcessFeatureEngineer()
        self.prod_model = None
        self.scaler = None
        self.pressure_surrogates = None
        self.best_model_name = "Random Forest"
        self.model_metrics = {}
        self.feature_names = FEATURE_COLUMNS
        self._load_trained_models()

    def _load_trained_models(self) -> None:
        """
        Loads serialized surrogate production model, pressure models, and scaler.
        """
        ensure_directories_exist()
        prod_model_path = MODELS_DIR / "choke_production_model.joblib"
        scaler_path = MODELS_DIR / "feature_scaler.joblib"
        pressure_path = MODELS_DIR / "pressure_surrogates.joblib"
        meta_path = MODELS_DIR / "model_metadata.joblib"

        if prod_model_path.exists() and scaler_path.exists():
            self.prod_model = joblib.load(prod_model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Successfully loaded surrogate production prediction model and scaler.")
        else:
            logger.warning("Trained model files not found. Controller running in physics-based dynamic fallback mode.")

        if pressure_path.exists():
            self.pressure_surrogates = joblib.load(pressure_path)
            logger.info("Loaded auxiliary pressure state surrogate predictors.")

        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.best_model_name = meta.get("best_model_name", "Surrogate Model")
            self.model_metrics = meta.get("metrics", {})

    def generate_candidate_actions(self, current_choke: float) -> np.ndarray:
        """
        Generates candidate choke positions within rate limit window:
        u ∈ [current_choke - 5.0%, current_choke + 5.0%] subject to [0%, 100%].

        Returns:
            np.ndarray: Array of candidate choke values.
        """
        max_delta = OPERATIONAL_CONSTRAINTS["MAX_DELTA_CHOKE"]
        choke_min = OPERATIONAL_CONSTRAINTS["CHOKE_MIN"]
        choke_max = OPERATIONAL_CONSTRAINTS["CHOKE_MAX"]

        lower_bound = max(choke_min, current_choke - max_delta)
        upper_bound = min(choke_max, current_choke + max_delta)

        candidates = np.arange(lower_bound, upper_bound + self.step_size / 2.0, self.step_size)
        return np.round(candidates, 2)

    def recommend_choke_position(self, current_state: Dict[str, float], prev_choke: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes single closed-loop control step to compute optimal choke position.

        Args:
            current_state: Dictionary containing sensor telemetry (Choke_Position, WHP, FLP, BHP, Oil_Rate).
            prev_choke: Choke position in previous interval (for oscillation dampening).

        Returns:
            Dict containing recommended choke position, expected outputs, safety status, XAI explanation, and audit trail.
        """
        current_choke = float(current_state["Choke_Position"])
        prev_choke_val = float(prev_choke) if prev_choke is not None else current_choke

        # 1. Candidate Generation
        candidates = self.generate_candidate_actions(current_choke)

        # 2. Dynamic Process Simulation & Telemetry Propagation
        pred_oil, pred_whp, pred_flp, pred_bhp = self._simulate_dynamic_process_candidates(candidates, current_state)

        # 3. Optimization & Constraint Evaluation
        df_candidates = self.optimizer.evaluate_candidates(
            candidates, current_state, prev_choke_val, pred_oil, pred_whp, pred_flp, pred_bhp
        )

        # 4. Decision Selection Logic
        safe_candidates = df_candidates[df_candidates["Is_Safe"] == True]

        if not safe_candidates.empty:
            best_choice = safe_candidates.iloc[0]
            status = "OPTIMAL_SAFE"
        else:
            best_choice = df_candidates.sort_values(by=["Violation_Count", "Delta_Choke"]).iloc[0]
            status = "SAFE_FALLBACK"

        recommended_choke = float(best_choice["Candidate_Choke"])
        choke_delta = round(recommended_choke - current_choke, 2)
        expected_oil_rate = float(best_choice["Predicted_Oil_Rate"])
        expected_whp = float(best_choice["Predicted_WHP"])
        expected_flp = float(best_choice["Predicted_FLP"])
        expected_bhp = float(best_choice["Predicted_BHP"])

        # 5. Explainable AI (XAI) Rationale & Model Confidence Calculation
        xai_explanation = self._generate_xai_explanation(
            current_state, recommended_choke, choke_delta, expected_oil_rate,
            expected_whp, expected_flp, expected_bhp, status, best_choice
        )
        
        confidence_score = self._calculate_model_confidence(df_candidates)

        return {
            "current_choke": current_choke,
            "recommended_choke": recommended_choke,
            "choke_delta": choke_delta,
            "expected_oil_rate": expected_oil_rate,
            "expected_whp": expected_whp,
            "expected_flp": expected_flp,
            "expected_bhp": expected_bhp,
            "status": status,
            "recommendation_reason": xai_explanation,
            "confidence_score": confidence_score,
            "best_model_used": self.best_model_name,
            "candidates_evaluated": len(candidates),
            "audit_trail": df_candidates
        }

    def _simulate_dynamic_process_candidates(
        self,
        candidates: np.ndarray,
        current_state: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates dynamic closed-loop process behavior for all candidate choke positions.
        Propagates choke movement relative to current telemetry inputs (WHP, FLP, BHP, Oil_Rate).
        """
        n = len(candidates)
        choke_base = float(current_state["Choke_Position"])
        whp_base = float(current_state["Wellhead_Pressure"])
        flp_base = float(current_state["Flowline_Pressure"])
        bhp_base = float(current_state["Bottom_Hole_Pressure"])
        oil_base = float(current_state.get("Oil_Rate", 600.0))

        delta_u = candidates - choke_base

        # 1. Hydrodynamic Pressure Response relative to current telemetry inputs
        # Opening choke (+delta_u) reduces WHP and BHP while increasing FLP
        sim_whp = whp_base - 3.5 * delta_u - 0.10 * (delta_u ** 2)
        sim_flp = flp_base + 0.9 * delta_u + 0.03 * (delta_u ** 2)
        sim_bhp = bhp_base - 5.0 * delta_u - 0.12 * (delta_u ** 2)

        # 2. Dynamic Inflow Performance & Choke Flow Capacity Response for Oil Production
        # Opening choke (+delta_u) increases flow capacity (+4.2 bbl/hr per 1% choke) with saturation curvature
        oil_hydro_delta = 4.5 * delta_u - 0.15 * (delta_u ** 2)

        if self.prod_model is not None and self.scaler is not None:
            feat_rows = []
            for i, u_cand in enumerate(candidates):
                f_dict = self.feature_engineer.transform_single_candidate(
                    current_state, u_cand, sim_whp[i], sim_flp[i], sim_bhp[i]
                )
                feat_rows.append(f_dict)

            df_feat = pd.DataFrame(feat_rows)[self.feature_names]
            X_scaled = self.scaler.transform(df_feat)
            ml_pred = self.prod_model.predict(X_scaled)
            
            # Anchor predictions relative to current oil_base while incorporating ML deltas and hydrodynamic choke flow sensitivity
            base_idx = len(candidates) // 2
            ml_delta = ml_pred - ml_pred[base_idx]
            sim_oil = oil_base + oil_hydro_delta + 0.3 * ml_delta
        else:
            sim_oil = oil_base + oil_hydro_delta

        sim_oil = np.maximum(sim_oil, 0.0)
        return sim_oil, sim_whp, sim_flp, sim_bhp




    def _generate_xai_explanation(
        self,
        current_state: Dict[str, float],
        rec_choke: float,
        choke_delta: float,
        exp_oil: float,
        exp_whp: float,
        exp_flp: float,
        exp_bhp: float,
        status: str,
        best_choice: pd.Series
    ) -> str:
        """
        Generates human-readable, industrial Explainable AI (XAI) rationale.
        """
        curr_choke = current_state["Choke_Position"]
        curr_oil = current_state.get("Oil_Rate", exp_oil)
        oil_diff = exp_oil - curr_oil
        oil_pct = (oil_diff / (curr_oil + 1e-4)) * 100.0

        if status == "OPTIMAL_SAFE":
            if abs(choke_delta) < 0.1:
                return f"Choke maintained at {rec_choke:.1f}% as current operating point optimizes production ({exp_oil:.1f} bbl/hr) within safe pressure envelopes."
            elif choke_delta > 0:
                return (f"Recommended choke opening increased by +{choke_delta:.1f}% (to {rec_choke:.1f}%) because predicted oil rate increases "
                        f"by +{oil_diff:.1f} bbl/hr (+{oil_pct:.1f}%) while maintaining safe WHP ({exp_whp:.1f} psi >= 250 psi) and FLP ({exp_flp:.1f} psi <= 450 psi).")
            else:
                return (f"Recommended choke opening reduced by {choke_delta:.1f}% (to {rec_choke:.1f}%) to protect wellhead backpressure "
                        f"and avoid pressure envelope violations, stabilizing WHP at {exp_whp:.1f} psi.")

        else:
            violations = best_choice.get("Violation_Details", "Operating near limit")
            return (f"Operating near safety boundaries. Recommended choke set to {rec_choke:.1f}% to minimize envelope violation risk. "
                    f"Audit note: {violations}.")

    def _calculate_model_confidence(self, df_candidates: pd.DataFrame) -> float:
        """
        Calculates controller decision confidence percentage [0% - 100%].
        """
        safe_cands = df_candidates[df_candidates["Is_Safe"] == True]
        if safe_cands.empty:
            return 65.0
        
        top_score = safe_cands.iloc[0]["Objective_Score"]
        avg_score = safe_cands["Objective_Score"].mean()
        
        if len(safe_cands) > 1:
            margin = top_score - avg_score
            confidence = min(98.5, max(75.0, 85.0 + margin * 0.5))
        else:
            confidence = 88.0

        return round(float(confidence), 1)

def compute_choke_recommendation(current_state: Dict[str, float], prev_choke: Optional[float] = None) -> Dict[str, Any]:
    """
    Convenience wrapper to instantiate AutonomousChokeController and get recommendation.
    """
    controller = AutonomousChokeController()
    return controller.recommend_choke_position(current_state, prev_choke=prev_choke)

