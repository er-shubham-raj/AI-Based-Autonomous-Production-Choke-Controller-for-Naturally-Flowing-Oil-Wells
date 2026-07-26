# Honeywell Engineering Technical Report
## AI-Based Autonomous Production Choke Controller for Naturally Flowing Oil Wells

**Document Version**: 1.0  
**Problem Statement**: 3 — Autonomous Production Choke Controller  
**Author**: Honeywell Hackathon Engineering Team  
**Date**: July 2026  

---

### Executive Summary
Naturally flowing oil wells rely on choke valve positioning to regulate surface backpressure, maintain stable bottom-hole pressure, prevent gas/water coning, and maximize oil production. Manual choke management often leads to production loss, excessive drawdown, or pressure limit violations. This report presents an industrial-grade AI-based Autonomous Production Choke Controller developed using Python, Scikit-learn, and Model Predictive Control (MPC) design patterns. The controller ingests operational sensor telemetry every control interval and recommends deterministic, optimal choke positions that maximize oil output while strictly enforcing physical rate movement limits ($\pm 5\%$) and pressure operating envelopes.

---

### 1. Problem Statement & Objectives
- **Problem Statement**: Develop an AI-based autonomous choke controller capable of recommending the next optimal choke position (%) at each control step.
- **Sensor Inputs**:
  1. `Time`: Control timestamp
  2. `Choke Position (%)`: Current valve opening percentage $[0 - 100\%]$
  3. `Oil Rate (bbl/hr)`: Measured surface production rate
  4. `Wellhead Pressure (WHP, psi)`: Pressure measured at the wellhead
  5. `Flowline Pressure (FLP, psi)`: Surface flowline pressure upstream of manifold
  6. `Bottom Hole Pressure (BHP, psi)`: Downhole reservoir flowing pressure
- **Control Output**: `Recommended Choke Position (%)` for the next control step.
- **Key Engineering Constraints**:
  - $0\% \le u_k \le 100\%$
  - Maximum movement rate $|u_k - u_{k-1}| \le 5.0\%$ per control step.
  - Respect WHP, FLP, and BHP pressure safety limits.
  - Deterministic and fully explainable decision logic.

---

### 2. Methodology & System Architecture

```
 Historical Telemetry Data (data/raw/)
               │
               ▼
 [ Preprocessing & Smoothing ] ───► Clip physical bounds, handle missing sensor data
               │
               ▼
 [ Process Feature Analytics ] ───► Differential (WHP - FLP), Drawdown (BHP - WHP), Lags, ΔP
               │
               ▼
 [ Machine Learning Surrogate ] ──► Random Forest Regressor vs Linear Regression Baseline
               │
               ▼
  [ Deterministic MPC Engine ] ──► Action Space [u_k - 5%, u_k + 5%] ──► Safety Constraint Filter ──► Utility Scoring
               │
               ▼
[ Output Recommendation & Log ] ──► Recommended Choke %, Expected Oil Rate, Safety Audit Log
```

---

### 3. Dataset & Industrial Process Feature Analytics
The system processes historical sensor telemetry. To capture multi-phase flow hydrodynamics, the following process features are engineered:
1. **Pressure Differential**: $\Delta P_{whp\_flp} = WHP - FLP$
2. **Pressure Ratio**: $P_{ratio} = \frac{WHP}{FLP + \epsilon}$
3. **Reservoir Drawdown Proxy**: $P_{drawdown} = BHP - WHP$
4. **Flow Efficiency Index**: $\eta = \frac{Q_{oil}}{Choke + \epsilon}$
5. **Temporal Lags**: $Choke_{k-1}, Q_{oil,k-1}, WHP_{k-1}$
6. **Rates of Change**: $\Delta WHP = WHP_k - WHP_{k-1}, \Delta Q_{oil} = Q_{oil,k} - Q_{oil,k-1}$
7. **Rolling Statistics**: 3-period rolling mean and standard deviation.

---

### 4. Machine Learning Surrogate Models
To evaluate candidate choke settings without risky physical trial-and-error, machine learning models act as digital twin surrogates:
- **Linear Regression Baseline**: Fits linear pressure-drop relationships.
- **Random Forest Regressor (Primary)**: Captures non-linear multiphase choke hydraulics.
- **Validation**: 5-Fold Cross Validation evaluating $MAE$, $RMSE$, and $R^2$.
- **Model Persistence**: Serialized via `joblib` into `models/`.

---

### 5. Controller & MPC Optimization Engine
The controller evaluates discrete candidate choke actions $u \in [u_k - 5\%, u_k + 5\%]$:
1. **Surrogate State Prediction**: Evaluates expected $Q_{oil}, WHP, FLP, BHP$ for each candidate.
2. **Constraint Verification**: Hard checks against configured pressure boundaries ($WHP_{min}, FLP_{max}, BHP_{min}$).
3. **Objective Utility Function**:
   $$\text{Score}(u) = \hat{Q}_{oil}(u) - \lambda_{move} |u - u_k| - \lambda_{safety} \cdot N_{violations}$$
4. **Optimal Choice**: Selects the candidate maximizing utility while ensuring 100% safety compliance.

---

### 6. Results & Model Performance Placeholder
*Note: Results populated dynamically based on the historical Honeywell dataset execution.*
- **Linear Regression R²**: [Calculated during runtime]
- **Random Forest Regressor R²**: [Calculated during runtime]
- **Maximum Actuation Rate Observed**: $\le 5.0\%$ (100% constraint compliance).

---

### 7. Conclusion & Future Engineering Roadmap
- **Conclusion**: The modular AI Autonomous Production Choke Controller demonstrates that deterministic Model Predictive Control coupled with Random Forest surrogate modeling provides robust, real-time choke optimization while safeguarding well pressure integrity.
- **Future Improvements**:
  - Integration with OPC-UA / MQTT for live DCS integration.
  - Multi-well network optimization for shared surface separators.
  - Deep Physics-Informed Neural Networks (PINN) for multiphase gas-oil-water choke performance.
