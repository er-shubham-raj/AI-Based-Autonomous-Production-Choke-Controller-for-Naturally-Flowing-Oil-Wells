"""
Model Training Module for Honeywell Autonomous Production Choke Controller.

Trains, evaluates, and benchmarks 4 surrogate prediction models:
1. Linear Regression (Industrial Baseline)
2. Random Forest Regressor
3. Extra Trees Regressor
4. Gradient Boosting Regressor

Evaluates performance using 5-Fold Cross Validation, computes MAE, RMSE, and R² scores,
extracts Feature Importances, generates a comparison benchmark table, auto-selects the best model,
and serializes models and scaler artifacts via Joblib.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_CONFIG, FEATURE_COLUMNS, MODELS_DIR
from src.utils import logger, calculate_metrics, ensure_directories_exist, set_seed

class ProductionSurrogateTrainer:
    """
    Trains and evaluates ML surrogate models for production and pressure state predictions.
    """

    def __init__(self, target_col: str = "Oil_Rate"):
        """
        Initialize Model Trainer.

        Args:
            target_col: Target variable name to predict (default: 'Oil_Rate').
        """
        self.target_col = target_col
        self.seed = MODEL_CONFIG["RANDOM_SEED"]
        set_seed(self.seed)
        ensure_directories_exist()
        
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: str = ""
        self.feature_importance_df: pd.DataFrame = pd.DataFrame()
        self.comparison_df: pd.DataFrame = pd.DataFrame()

    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Executes complete 4-model training, cross-validation, feature importance extraction,
        comparison table generation, and model serialization pipeline.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple containing dictionary of evaluation results and feature importance DataFrame.
        """
        logger.info(f"Preparing dataset for target variable: '{self.target_col}'...")
        
        # Verify available features present in dataframe
        available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
        X = df[available_features]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG["TEST_SIZE"], random_state=self.seed, shuffle=False
        )

        logger.info(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Instantiate 4 Regressors
        candidate_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=MODEL_CONFIG.get("RF_N_ESTIMATORS", 150),
                max_depth=MODEL_CONFIG.get("RF_MAX_DEPTH", 12),
                random_state=self.seed,
                n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=MODEL_CONFIG.get("ET_N_ESTIMATORS", 150),
                max_depth=MODEL_CONFIG.get("ET_MAX_DEPTH", 12),
                random_state=self.seed,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG.get("GB_N_ESTIMATORS", 150),
                max_depth=MODEL_CONFIG.get("GB_MAX_DEPTH", 6),
                learning_rate=MODEL_CONFIG.get("GB_LEARNING_RATE", 0.05),
                random_state=self.seed
            )
        }

        kf = KFold(n_splits=MODEL_CONFIG["CV_FOLDS"], shuffle=True, random_state=self.seed)
        comparison_rows = []

        logger.info("Executing 4-Model Benchmark Training & Cross-Validation...")

        for name, model in candidate_models.items():
            logger.info(f"Training Model: '{name}'...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = calculate_metrics(y_test.values, y_pred)

            # 5-Fold Cross Validation
            cv_scores = cross_validate(
                model, X_train_scaled, y_train, cv=kf,
                scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
            )
            cv_r2_mean = round(float(np.mean(cv_scores['test_r2'])), 4)
            cv_mae_mean = round(float(abs(np.mean(cv_scores['test_neg_mean_absolute_error']))), 4)

            metrics["CV_R2_Mean"] = cv_r2_mean
            metrics["CV_MAE_Mean"] = cv_mae_mean

            self.models[name] = model
            self.results[name] = metrics

            comparison_rows.append({
                "Model": name,
                "R2_Score": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "CV_R2_Mean": cv_r2_mean,
                "CV_MAE_Mean": cv_mae_mean
            })

            logger.info(f"[{name}] -> R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")

        # Comparison DataFrame
        self.comparison_df = pd.DataFrame(comparison_rows).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)
        
        # Auto-Select Best Model by highest R2 Score
        self.best_model_name = self.comparison_df.iloc[0]["Model"]
        logger.info(f"\n🏆 Best Model Selected: '{self.best_model_name}' (R² = {self.comparison_df.iloc[0]['R2_Score']:.4f})\n")

        # Feature Importance Analysis from best model (or ensemble models)
        best_model = self.models[self.best_model_name]
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(self.models.get("Random Forest"), "feature_importances_"):
            importances = self.models["Random Forest"].feature_importances_
        else:
            importances = np.abs(getattr(best_model, "coef_", np.ones(len(available_features))))

        self.feature_importance_df = pd.DataFrame({
            "Feature": available_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Save trained artifacts
        self.save_artifacts(available_features)

        # Train state pressure surrogate predictors (WHP, FLP, BHP) for dynamic MPC
        self._train_state_pressure_predictors(df, available_features)

        return self.results, self.feature_importance_df

    def _train_state_pressure_predictors(self, df: pd.DataFrame, features: List[str]) -> None:
        """
        Trains multi-channel surrogate models for predicting WHP, FLP, and BHP states given choke adjustments.
        """
        logger.info("Training auxiliary pressure surrogate state predictors (WHP, FLP, BHP)...")
        X = df[features]
        X_scaled = self.scaler.transform(X)

        pressure_models = {}
        for pres_target in ["Wellhead_Pressure", "Flowline_Pressure", "Bottom_Hole_Pressure"]:
            if pres_target in df.columns:
                y_pres = df[pres_target]
                model = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1)
                model.fit(X_scaled, y_pres)
                pressure_models[pres_target] = model
                logger.info(f"Trained pressure surrogate for: '{pres_target}'")

        joblib.dump(pressure_models, MODELS_DIR / "pressure_surrogates.joblib")
        logger.info("Auxiliary pressure surrogate models saved.")

    def save_artifacts(self, feature_list: List[str]) -> None:
        """
        Saves best model, all candidate models, scaler, and feature metadata via Joblib.
        """
        best_model = self.models[self.best_model_name]
        model_path = MODELS_DIR / "choke_production_model.joblib"
        all_models_path = MODELS_DIR / "all_trained_models.joblib"
        scaler_path = MODELS_DIR / "feature_scaler.joblib"
        meta_path = MODELS_DIR / "model_metadata.joblib"

        joblib.dump(best_model, model_path)
        joblib.dump(self.models, all_models_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            "best_model_name": self.best_model_name,
            "metrics": self.results,
            "comparison": self.comparison_df.to_dict(orient="records"),
            "feature_names": feature_list
        }, meta_path)

        logger.info(f"Model artifacts successfully saved to: {MODELS_DIR}")

def train_surrogate_models(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Convenience function to run full 4-model training pipeline.
    """
    trainer = ProductionSurrogateTrainer()
    return trainer.train_and_evaluate(df)

