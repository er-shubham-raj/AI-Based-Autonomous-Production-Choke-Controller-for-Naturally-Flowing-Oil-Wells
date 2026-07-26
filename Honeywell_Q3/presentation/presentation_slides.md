# Honeywell Hackathon Submission — Problem Statement 3
## AI-Based Autonomous Production Choke Controller for Naturally Flowing Oil Wells

---

## Slide 1: Title & Executive Summary
- **Project Title**: AI-Based Autonomous Production Choke Controller for Naturally Flowing Oil Wells
- **Domain**: Industrial Automation & Oil/Gas Digitalization
- **Team**: Honeywell Hackathon Engineering Team
- **Objective**: Deliver a production-ready, deterministic Model Predictive Control (MPC) surrogate system that optimizes choke opening every control interval to maximize oil recovery while respecting wellhead, flowline, and bottom-hole pressure safety envelopes.

---

## Slide 2: Proposed Solution
- **Industrial Challenge**: Manual choke adjustment leads to suboptimal production rates, severe liquid loading risk, or excessive reservoir pressure drawdown.
- **Autonomous Solution**: An AI-surrogate Model Predictive Controller (MPC) that ingests real-time sensor telemetry (Time, Choke %, Oil Rate bbl/hr, WHP, FLP, BHP).
- **Core Mechanism**:
  1. Dynamic Action Space Candidate Generation ($u \in [u_k - 5\%, u_k + 5\%]$).
  2. Multi-channel Machine Learning State Predictor ($Q_{oil}, WHP, FLP, BHP$).
  3. Safety Constraint Matrix (Hard boundary filtering).
  4. Multi-Objective Utility Engine (Maximizes production, penalizes pressure risk & choke hunting).
- **Key Value**: Fully explainable, deterministic, zero black-box risk.

---

## Slide 3: Technical Architecture
- **Data & Feature Layer**: Process feature analytics derived from telemetry:
  - Pressure Differential ($WHP - FLP$) & Ratio ($WHP / FLP$)
  - Drawdown Proxy ($BHP - WHP$) & Flow Efficiency ($\eta = Q_{oil} / Choke$)
  - Temporal Lags ($Choke_{k-1}, Q_{oil,k-1}$) & Rolling Noise Filtered Trends
- **Surrogate Engine**:
  - **Baseline**: Linear Regression for linear hydrodynamic response.
  - **Primary Model**: Random Forest Regressor capturing complex multiphase flow dynamics.
- **Decision Engine (Simplified MPC)**:
  - Step resolution discretization ($0.5\%$).
  - Pressure envelope safety verification ($WHP, FLP, BHP$ boundaries).
  - Deterministic scoring function & audit log generation.

---

## Slide 4: Engineering Feasibility & Safety Compliance
- **Strict Constraints Enforced**:
  - Choke valve travel bounds: $0\% \le u_k \le 100\%$
  - Maximum movement limit: $|u_k - u_{k-1}| \le 5\%$ per control step.
  - Pressure limits: Configurable $WHP_{min}, FLP_{max}, BHP_{min}$ limits.
- **Deterministic & Safe**: Eliminates unconstrained Reinforcement Learning instability. Fallback logic defaults to minimum violation hold if well approaches emergency shut-in boundaries.
- **Low Latency**: Inference and candidate scoring executed in $< 15\text{ ms}$, fully compatible with standard Industrial IoT and Honeywell Experion® DCS control loops.

---

## Slide 5: Project Artifacts & Deliverables
- **Modular Python Architecture**: Clean package structure (`src/`, `data/`, `models/`, `plots/`, `notebooks/`).
- **Jupyter Notebook Suite**:
  - `01_EDA.ipynb`: Sensor telemetry diagnostics.
  - `02_Model_Training.ipynb`: Feature engineering, cross-validation, and model serialization.
  - `03_Controller_Testing.ipynb`: Closed-loop backtest simulation & constraint audit.
- **Honeywell Dark Dashboard**: Streamlit interactive control panel (`dashboard/app.py`).
- **Automated Visualization Engine**: 9 process charts exported automatically to `plots/`.
- **Engineering Report**: Comprehensive industrial technical documentation (`report/engineering_report.md`).

---

## Slide 6: References & Industry Standards
- **Honeywell Experion® PKS**: Distributed Control System (DCS) & Advanced Process Control (APC) principles.
- **Nodal Analysis for Well Performance**: Beggs & Brill Multiphase Flow Correlations.
- **Model Predictive Control (MPC)**: Camacho, E. F., & Alba, C. B. (2013). *Model Predictive Control*. Springer Science & Business Media.
- **Python Industrial Ecosystem**: Pandas, Scikit-Learn, Plotly, Joblib, Streamlit.
