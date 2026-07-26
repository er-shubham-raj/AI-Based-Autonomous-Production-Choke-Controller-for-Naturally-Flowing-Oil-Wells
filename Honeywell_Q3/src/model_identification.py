"""
Dynamic Model Identification Module for Honeywell Autonomous Production Choke Controller.

Uses open-loop step test data to fit a First Order Plus Dead Time (FOPDT) dynamic transfer function model:
G(s) = (Kp / (tau * s + 1)) * exp(-theta * s)

Estimates Process Gain Kp, Time Constant tau, Dead Time theta, and evaluates model fit (R², RMSE).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

from src.utils import logger, calculate_metrics

class ProcessModelIdentifier:
    """
    Identifies FOPDT dynamic process model from step-test time-series data.
    """

    def __init__(self):
        pass

    @staticmethod
    def fopdt_response(t: np.ndarray, y0: float, Kp: float, du: float, tau: float, theta: float, t_step: float = 1.25) -> np.ndarray:
        """
        Analytical First Order Plus Dead Time step response function.
        """
        y_pred = np.full_like(t, y0)
        t_effective = t - t_step - theta
        mask = t_effective > 0.0
        y_pred[mask] = y0 + Kp * du * (1.0 - np.exp(-t_effective[mask] / max(tau, 1e-3)))
        return y_pred

    def identify_fopdt(self, df_step: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits FOPDT dynamic parameters (Kp, tau, theta) on a step test DataFrame.

        Args:
            df_step: DataFrame containing step test time series (Time_Min, Oil_Rate, Choke_Position).

        Returns:
            Dict containing identified parameters, prediction arrays, and fit error metrics.
        """
        t = df_step["Time_Min"].values
        y_meas = df_step["Oil_Rate"].values
        u = df_step["Choke_Position"].values

        y0 = float(y_meas[0])
        u0 = float(u[0])
        u_final = float(u[-1])
        du = u_final - u0
        t_step = float(t[np.where(u != u0)[0][0]]) if len(np.where(u != u0)[0]) > 0 else 1.25

        # Analytical initial guess
        y_ss = float(y_meas[-1])
        kp_guess = (y_ss - y0) / (du + 1e-5)
        tau_guess = 2.0
        theta_guess = 0.2

        try:
            # Objective wrapper for curve_fit
            def fit_func(t_val, Kp, tau, theta):
                return self.fopdt_response(t_val, y0, Kp, du, abs(tau), abs(theta), t_step=t_step)

            popt, _ = curve_fit(
                fit_func, t, y_meas,
                p0=[kp_guess, tau_guess, theta_guess],
                bounds=([-50.0, 0.1, 0.0], [50.0, 30.0, 10.0])
            )
            kp_fit, tau_fit, theta_fit = popt
        except Exception as e:
            logger.warning(f"FOPDT curve fitting fallback to analytical estimates: {e}")
            kp_fit = kp_guess
            tau_fit = 2.5
            theta_fit = 0.25

        y_pred = self.fopdt_response(t, y0, kp_fit, du, tau_fit, theta_fit, t_step=t_step)

        metrics = calculate_metrics(y_meas, y_pred)

        engineering_explanation = (
            f"Identified First Order Plus Dead Time (FOPDT) dynamic model: "
            f"Process Gain Kp = {kp_fit:.2f} bbl/hr per % choke, Time Constant τ = {tau_fit:.2f} min, "
            f"Dead Time θ = {theta_fit:.2f} min. Model fitting achieved R² = {metrics['R2']:.4f} and RMSE = {metrics['RMSE']:.2f} bbl/hr."
        )

        return {
            "Kp": round(float(kp_fit), 4),
            "tau_min": round(float(tau_fit), 2),
            "theta_min": round(float(theta_fit), 2),
            "y0": y0,
            "du": du,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "y_measured": y_meas.tolist(),
            "y_predicted": y_pred.tolist(),
            "time_min": t.tolist(),
            "engineering_explanation": engineering_explanation
        }

def identify_dynamic_process_model(df_step: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience wrapper to identify FOPDT model parameters.
    """
    identifier = ProcessModelIdentifier()
    return identifier.identify_fopdt(df_step)
