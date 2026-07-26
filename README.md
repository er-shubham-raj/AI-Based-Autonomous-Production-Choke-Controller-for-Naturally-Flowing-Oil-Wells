# AI-Based Autonomous Production Choke Controller for Naturally Flowing Oil Wells
### Honeywell Hackathon — Problem Statement 3 Submission

![Honeywell Banner](https://img.shields.io/badge/Honeywell-Process%20Solutions-red?style=for-the-badge)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge)

---

## 📌 Project Overview
In naturally flowing oil wells, the **choke valve** is the primary control device used to regulate surface flow rates, manage wellhead backpressure, maintain bottom-hole pressure above bubble point, and prevent liquid loading or gas coning.

This repository provides an **industrial-grade, deterministic AI control system** built in accordance with Honeywell engineering principles. It pairs process feature analytics and surrogate machine learning models with a **Model Predictive Control (MPC)** engine to recommend optimal choke positions every control interval while maintaining strict pressure safety envelopes.

---

## ⚙️ Key Engineering Constraints Enforced
1. **Choke Operational Bounds**: $0\% \le \text{Choke} \le 100\%$
2. **Maximum Actuation Speed**: $|\Delta \text{Choke}| = |u_k - u_{k-1}| \le 5.0\%$ per control interval.
3. **Pressure Operating Envelope**:
   - Minimum safe Wellhead Pressure ($WHP_{min} \ge 250\text{ psi}$)
   - Maximum safe Flowline Pressure ($FLP_{max} \le 450\text{ psi}$)
   - Bottom Hole Pressure Drawdown limit ($BHP_{min} \ge 800\text{ psi}$)
4. **Determinism & Explainability**: 100% explainable candidate scoring matrix with zero black-box optimization risk.

---

## 📁 Repository Structure

```
Honeywell_Q3/
│
├── data/
│   ├── raw/                   # Historical operational dataset (CSV)
│   └── processed/             # Preprocessed & feature-engineered dataset
│
├── models/                    # Serialized ML models (.joblib) & scalers
├── plots/                     # Automatically exported high-res visualizations
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Sensor telemetry diagnostics & correlation
│   ├── 02_Model_Training.ipynb # Feature engineering, cross-validation & ML training
│   └── 03_Controller_Testing.ipynb # Closed-loop MPC simulation & constraint verification
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Lightweight configuration parameters
│   ├── data_loader.py         # CSV data loader for historical telemetry
│   ├── preprocessing.py       # Sensor noise clipping & missing value handling
│   ├── feature_engineering.py # Process analytics (pressure diffs, ratios, lags, rates)
│   ├── train_model.py         # Linear Regression baseline & Random Forest model pipeline
│   ├── optimizer.py           # Multi-objective score engine & constraint checker
│   ├── controller.py          # Simplified MPC Autonomous Choke Controller engine
│   ├── visualization.py       # Automated industrial plotting module
│   └── utils.py               # Logger, metrics computation (MAE, RMSE, R²), seeds
│
├── dashboard/
│   └── app.py                 # Streamlit dark-themed industrial control panel
│
├── presentation/
│   └── presentation_slides.md # 6-Slide presentation deck matching hackathon template
│
├── report/
│   └── engineering_report.md  # Comprehensive industrial engineering report
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & quickstart guide
└── main.py                    # End-to-end execution entry point
```

---

## 🚀 Quickstart & Installation Guide

### 1. Prerequisites
Ensure Python **3.11** is installed on your system.

### 2. Install Dependencies
Navigate to the project folder and install required packages:
```bash
cd Honeywell_Q3
pip install -r requirements.txt
```

### 3. Run End-to-End Pipeline
Execute `main.py` to run the full workflow (data loading, preprocessing, feature engineering, surrogate ML training, plot generation, report verification, and controller backtest simulation):
```bash
python main.py
```

### 4. Launch Streamlit Industrial Control Dashboard
Launch the dark-themed Honeywell dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 🧮 Process Feature Engineering & MPC Formulation

### Industrial Feature Engineering
- **Pressure Differential**: $\Delta P = WHP - FLP$
- **Pressure Ratio**: $P_{ratio} = \frac{WHP}{FLP + \epsilon}$
- **Drawdown Proxy**: $P_{drawdown} = BHP - WHP$
- **Flow Efficiency**: $\eta = \frac{Q_{oil}}{\text{Choke} + \epsilon}$
- **Temporal Lags**: $\text{Choke}_{k-1}, Q_{oil,k-1}, WHP_{k-1}$
- **Rates of Change**: $\Delta WHP, \Delta Q_{oil}$

### MPC Objective Utility Scoring
For candidate choke openings $u \in [u_k - 5\%, u_k + 5\%]$:
$$\text{Score}(u) = \hat{Q}_{oil}(u) - \lambda_{move} |u - u_k| - \lambda_{safety} \cdot N_{violations}$$

---

## 📊 Automatically Exported Visualizations
When running `main.py`, the following publication-quality process charts are exported to `plots/`:
1. `correlation_heatmap.png`: Sensor channel feature correlations.
2. `oil_production_trend.png`: Historical oil production trajectory.
3. `pressure_trends.png`: WHP, FLP, and BHP pressure envelope trends.
4. `choke_position_timeline.png`: Historical choke valve position trajectory.
5. `feature_importance.png`: Top surrogate feature importances.
6. `prediction_vs_actual.png`: ML model parity plot.
7. `model_comparison.png`: Linear Regression baseline vs Random Forest benchmark.
8. `controller_recommendation_timeline.png`: Autonomous MPC choke trajectory vs baseline.
9. `safety_constraint_status.png`: Safety constraint compliance breakdown.

---

## 🛠️ Configuration & Customization
Safety pressure boundaries, choke step sizes, and ML hyperparameter thresholds can be easily adjusted in `src/config.py`:
```python
OPERATIONAL_CONSTRAINTS = {
    "CHOKE_MIN": 0.0,
    "CHOKE_MAX": 100.0,
    "MAX_DELTA_CHOKE": 5.0,  # Max shift ±5%
    "WHP_MIN": 250.0,        # Min safe WHP (psi)
    "WHP_MAX": 1200.0,       # Max safe WHP (psi)
    "FLP_MIN": 50.0,
    "FLP_MAX": 450.0,
    "BHP_MIN": 800.0
}
```

---

## 📄 License & Hackathon Submission
Designed for the **Honeywell Hackathon Problem Statement 3**.
Built with industrial rigor, zero synthetic data dependencies, and deterministic safety controls.

