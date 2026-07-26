"""
Open-Loop Step Test Analysis Module for Honeywell Autonomous Production Choke Controller.

Simulates open-loop choke step experiments (30% -> 40%, 40% -> 50%, 50% -> 60%, 60% -> 70%),
records time-domain responses for Oil Rate, WHP, FLP, and BHP, and computes step response metrics
(Process Gain Kp, Settling Time Ts, Max Response, Steady State Value).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.utils import logger

class OpenLoopStepTester:
    """
    Executes open-loop step response tests on naturally flowing oil well simulator.
    """

    def __init__(self, time_steps: int = 50, dt_sec: float = 15.0):
        """
        Initialize Open-Loop Step Tester.

        Args:
            time_steps: Number of time steps per step test.
            dt_sec: Time step duration in seconds.
        """
        self.time_steps = time_steps
        self.dt_sec = dt_sec

    def run_step_test(self, initial_choke: float, target_choke: float, step_at: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes a single open-loop choke step test from initial_choke to target_choke.

        Returns:
            Tuple of (DataFrame time series, Summary metrics dict).
        """
        choke_shift = target_choke - initial_choke
        time_arr = np.arange(self.time_steps) * (self.dt_sec / 60.0)  # Time in minutes

        # Physical baseline values at initial_choke
        base_whp = 1000.0 - 5.0 * initial_choke
        base_flp = 150.0 + 1.2 * initial_choke
        base_bhp = 2500.0 - 8.0 * initial_choke
        base_oil = max(0.0, 12.0 * initial_choke - 0.08 * (initial_choke ** 1.5) + 0.1 * (base_whp - base_flp))

        # First order response parameters (tau = 2.5 min)
        tau = 2.5
        t_step = (step_at * self.dt_sec) / 60.0

        records = []
        for i, t in enumerate(time_arr):
            if t < t_step:
                u = initial_choke
                response_factor = 0.0
            else:
                u = target_choke
                response_factor = 1.0 - np.exp(-(t - t_step) / tau)

            # Dynamic hydrodynamic responses
            delta_u_effective = choke_shift * response_factor
            
            whp = base_whp - 3.5 * delta_u_effective - 0.08 * (delta_u_effective ** 2) + np.random.normal(0, 0.2)
            flp = base_flp + 0.9 * delta_u_effective + 0.02 * (delta_u_effective ** 2) + np.random.normal(0, 0.1)
            bhp = base_bhp - 5.0 * delta_u_effective - 0.10 * (delta_u_effective ** 2) + np.random.normal(0, 0.3)

            oil = base_oil + 4.5 * delta_u_effective - 0.12 * (delta_u_effective ** 2) + np.random.normal(0, 0.2)
            oil = max(0.0, oil)

            records.append({
                "Time_Min": round(t, 2),
                "Step_Name": f"{initial_choke:.0f}% -> {target_choke:.0f}%",
                "Choke_Position": round(u, 2),
                "Oil_Rate": round(oil, 2),
                "Wellhead_Pressure": round(whp, 2),
                "Flowline_Pressure": round(flp, 2),
                "Bottom_Hole_Pressure": round(bhp, 2)
            })

        df_step = pd.DataFrame(records)

        # Summary Metrics Calculation
        y_0 = df_step["Oil_Rate"].iloc[0]
        y_ss = df_step["Oil_Rate"].iloc[-1]
        delta_y = y_ss - y_0
        kp_gain = delta_y / (choke_shift + 1e-5)
        y_max = df_step["Oil_Rate"].max()

        # Settling time (time to reach 95% of steady state)
        target_95 = y_0 + 0.95 * delta_y
        settling_rows = df_step[df_step["Oil_Rate"] >= target_95]
        if not settling_rows.empty:
            settling_time_min = float(settling_rows.iloc[0]["Time_Min"]) - t_step
            settling_time_min = max(0.0, settling_time_min)
        else:
            settling_time_min = 4.0 * tau

        summary = {
            "Step_Experiment": f"{initial_choke:.0f}% -> {target_choke:.0f}%",
            "Initial_Choke": initial_choke,
            "Target_Choke": target_choke,
            "Choke_Step_Size": choke_shift,
            "Initial_Oil_Rate": round(y_0, 2),
            "Steady_State_Oil_Rate": round(y_ss, 2),
            "Max_Response_Oil": round(y_max, 2),
            "Response_Gain_Kp": round(kp_gain, 4),
            "Settling_Time_Min": round(settling_time_min, 2)
        }

        return df_step, summary

    def run_all_step_experiments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs standard hackathon step experiments:
        30% -> 40%, 40% -> 50%, 50% -> 60%, 60% -> 70%.

        Returns:
            Tuple of (Combined time series DataFrame, Summary metrics DataFrame).
        """
        step_pairs = [(30.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 70.0)]
        all_dfs = []
        all_summaries = []

        logger.info("Executing 4 Open-Loop Step Test Experiments...")

        for u0, u1 in step_pairs:
            df_s, sum_s = self.run_step_test(u0, u1)
            all_dfs.append(df_s)
            all_summaries.append(sum_s)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        summary_df = pd.DataFrame(all_summaries)

        logger.info(f"Open-Loop Step Tests complete. Generated {len(combined_df)} telemetry data points.")
        return combined_df, summary_df

def generate_step_test_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to run open-loop step test experiments.
    """
    tester = OpenLoopStepTester()
    return tester.run_all_step_experiments()
