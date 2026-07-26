"""
Honeywell Autonomous Production Choke Controller — Main Pipeline Execution Entry Point.

Executes the complete industrial workflow:
1. Historical Telemetry Loading
2. Data Preprocessing & Physical Range Filtering
3. Industrial Process Feature Analytics
4. ML Surrogate Model Training (Baseline vs Random Forest) & Cross Validation
5. Model Serialization (Joblib)
6. Autonomous MPC Controller Simulation & Constraint Audit
7. Automated Export of High-Resolution Visualizations
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import setup_logger, ensure_directories_exist, set_seed
from src.data_loader import load_historical_telemetry
from src.preprocessing import preprocess_telemetry
from src.feature_engineering import engineer_features
from src.train_model import train_surrogate_models
from src.controller import AutonomousChokeController
from src.visualization import generate_all_plots
from src.step_test import generate_step_test_analysis
from src.model_identification import identify_dynamic_process_model
from src.scenarios import run_challenge_scenarios

logger = setup_logger("Honeywell_Main_Pipeline")

def run_pipeline() -> None:
    """
    Runs the complete end-to-end execution pipeline.
    """
    logger.info("==========================================================================")
    logger.info("  HONEYWELL AUTONOMOUS PRODUCTION CHOKE CONTROLLER - PIPELINE STARTING    ")
    logger.info("==========================================================================")

    # 0. Environment Setup & Seed Initialization
    ensure_directories_exist()
    set_seed(42)

    # 1. Data Loading
    logger.info("\n[STEP 1/7] Loading Historical Sensor Telemetry Dataset...")
    df_raw = load_historical_telemetry()
    logger.info(f"Loaded dataset with {len(df_raw)} records.")

    # 2. Data Preprocessing
    logger.info("\n[STEP 2/7] Executing Preprocessing & Physical Envelope Filtering...")
    df_clean = preprocess_telemetry(df_raw)

    # 3. Process Feature Engineering
    logger.info("\n[STEP 3/7] Engineering Industrial Process Analytics Features...")
    df_feat = engineer_features(df_clean)

    # 4. Machine Learning Model Training & Evaluation
    logger.info("\n[STEP 4/7] Training Surrogate Machine Learning Models...")
    results, feature_imp_df = train_surrogate_models(df_feat)
    
    logger.info("--------------------------------------------------------------------------")
    logger.info("Model Benchmark Performance Metrics:")
    for model_name, metrics in results.items():
        logger.info(f"  > {model_name:20s} | R²: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.4f} bbl/hr | RMSE: {metrics['RMSE']:.4f} bbl/hr")
    logger.info("--------------------------------------------------------------------------")

    # 5. Open-Loop Step Tests & FOPDT Model Identification
    logger.info("\n[STEP 5/7] Executing Open-Loop Step Tests & FOPDT Model Identification...")
    step_df, step_summary = generate_step_test_analysis()
    step_40_50 = step_df[step_df["Step_Name"] == "40% -> 50%"]
    fopdt_fit = identify_dynamic_process_model(step_40_50)
    logger.info(f"FOPDT Identified: Kp = {fopdt_fit['Kp']:.2f} bbl/hr/%, tau = {fopdt_fit['tau_min']:.2f} min, theta = {fopdt_fit['theta_min']:.2f} min (R² = {fopdt_fit['R2']:.4f})")

    # 6. Autonomous Controller Simulation & Challenge Scenarios (A, B, C)
    logger.info("\n[STEP 6/7] Running Closed-Loop MPC Simulation & Challenge Scenarios (A, B, C)...")
    sc_summary, sc_dict = run_challenge_scenarios()
    logger.info("--------------------------------------------------------------------------")
    logger.info("Challenge Scenarios Performance Matrix:")
    for _, sc_row in sc_summary.iterrows():
        logger.info(f"  > {sc_row['Scenario']:32s} | Target: {sc_row['Target_Rate']:6.1f} | Final: {sc_row['Final_Rate']:6.1f} bbl/hr | Safe: {sc_row['Is_Safe']}")
    logger.info("--------------------------------------------------------------------------")

    controller = AutonomousChokeController()
    sim_records = []
    sample_steps = min(150, len(df_feat))
    prev_choke = None
    for i in range(sample_steps):
        row = df_feat.iloc[i]
        state = {
            "Choke_Position": float(row["Choke_Position"]),
            "Wellhead_Pressure": float(row["Wellhead_Pressure"]),
            "Flowline_Pressure": float(row["Flowline_Pressure"]),
            "Bottom_Hole_Pressure": float(row["Bottom_Hole_Pressure"]),
            "Oil_Rate": float(row["Oil_Rate"])
        }
        rec = controller.recommend_choke_position(state, prev_choke=prev_choke)
        prev_choke = rec["recommended_choke"]

        sim_records.append({
            "Step": i,
            "Current_Choke": rec["current_choke"],
            "Recommended_Choke": rec["recommended_choke"],
            "Choke_Delta": rec["choke_delta"],
            "Expected_Oil_Rate": rec["expected_oil_rate"],
            "Status": rec["status"],
            "Confidence_Score": rec["confidence_score"],
            "audit_trail": rec["audit_trail"]
        })

    sim_df = pd.DataFrame(sim_records)

    # 7. Export Visualizations
    logger.info("\n[STEP 7/7] Generating Publication-Quality Process Visualizations...")
    saved_plots = generate_all_plots(df_feat, results, feature_imp_df, sim_df)
    logger.info(f"Exported {len(saved_plots)} process plots to 'plots/' directory.")


    logger.info("\n==========================================================================")
    logger.info("  HONEYWELL PIPELINE EXECUTION COMPLETED SUCCESSFULLY WITH ZERO ERRORS    ")
    logger.info("==========================================================================")
    logger.info("Quick Links:")
    logger.info(" - Launch Dashboard:   streamlit run dashboard/app.py")
    logger.info(" - View Report:        report/engineering_report.md")
    logger.info(" - View Presentation:  presentation/presentation_slides.md")
    logger.info("==========================================================================")

if __name__ == "__main__":
    run_pipeline()


