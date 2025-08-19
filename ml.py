import logging
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, KFold


NON_FEATURE_COLS = {
    "LST_Cluster", "LST_Zone", "latitude", "longitude",
    "Gi_ZScore", "Gi_PValue", "Hot_Spot_Class", "Year",
}


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # FIX: "Season" is no longer a feature, so it's removed from the encoding step.
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    
    # One-hot encode only the Land Cover Type.
    df_ml = pd.get_dummies(df.drop(columns=cols_to_drop), columns=["Land_Cover_Type"], prefix="LC", dummy_na=False)
    
    if "LST_Celsius" not in df_ml.columns:
        raise ValueError("Target column 'LST_Celsius' missing for ML")
    
    X = df_ml.drop(columns=["LST_Celsius"])
    y = df_ml["LST_Celsius"]
    
    if X.empty or y.empty:
        raise ValueError("Empty features/target after preprocessing")
    
    logging.info("Features prepared for ML. Shape: %s. Features: %s", X.shape, list(X.columns))
    return X, y


# The rest of the file (train_random_forest, evaluate_and_plot, shap_analysis) remains unchanged.
# I am including it in full for completeness.

def train_random_forest(
    X: pd.DataFrame, y: pd.Series, random_state: int,
    perform_tuning: bool, tuning_iterations: int, default_params: Dict
) -> RandomForestRegressor:
    if perform_tuning:
        logging.info("Hyperparameter tuning (%d iterations)...", tuning_iterations)
        rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 300, 500], "max_features": ["sqrt", "log2", 1.0],
            "max_depth": [10, 15, 20, 25, None], "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4], "bootstrap": [True, False],
        }
        search = RandomizedSearchCV(
            estimator=rf, param_distributions=param_dist, n_iter=tuning_iterations, cv=5,
            verbose=1, random_state=random_state, n_jobs=-1, scoring="neg_root_mean_squared_error",
        )
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=random_state)
        search.fit(X_train, y_train)
        logging.info("Best params: %s", search.best_params_)
        return search.best_estimator_
    else:
        logging.info("Training RF with default parameters.")
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **default_params)
        model.fit(X, y)
        return model

def evaluate_and_plot(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series, outputs_dir: str, random_state: int) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    n = len(X_train)
    cv_splits = min(5, n) if n > 1 else 0
    if cv_splits > 1:
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
    else:
        cv_rmse, cv_r2 = np.array([]), np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ENHANCEMENT: Use descriptive labels
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=25, color="mediumblue")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0, 0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
    axes[0, 0].set_title(f"Model Predictions vs. Actual Temperatures (R²={r2:.3f})", fontsize=14)
    axes[0, 0].set_xlabel("Actual Temperature (°C)")
    axes[0, 0].set_ylabel("Predicted Temperature (°C)")

    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=25, color="darkgreen")
    axes[0, 1].axhline(0, color="r", linestyle="--")
    axes[0, 1].set_title("Prediction Errors (Residuals)", fontsize=14)
    axes[0, 1].set_xlabel("Predicted Temperature (°C)")
    axes[0, 1].set_ylabel("Prediction Error (°C)")
    
    importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    importances.plot(kind="barh", ax=axes[1, 0], color=sns.color_palette("viridis", len(importances)))
    axes[1, 0].set_title("Top 10 Most Important Features", fontsize=14)
    axes[1, 0].invert_yaxis()
    
    sns.histplot(residuals, bins=30, ax=axes[1, 1], color="goldenrod", kde=True)
    axes[1, 1].set_title("Distribution of Prediction Errors", fontsize=14)
    axes[1, 1].set_xlabel("Prediction Error (°C)")

    plt.tight_layout()
    path = os.path.join(outputs_dir, "model_evaluation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved model evaluation plots: %s", path)

    return {
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
        "cv_rmse_mean": float(np.mean(cv_rmse)) if cv_rmse.size > 0 else -1.0,
        "cv_rmse_std": float(np.std(cv_rmse)) if cv_rmse.size > 0 else -1.0,
        "cv_r2_mean": float(np.mean(cv_r2)) if cv_r2.size > 0 else -1.0,
        "cv_r2_std": float(np.std(cv_r2)) if cv_r2.size > 0 else -1.0,
    }

def shap_analysis(model: RandomForestRegressor, X: pd.DataFrame, outputs_dir: str, random_state: int) -> None:
    try:
        import shap
    except ImportError:
        logging.warning("SHAP library not installed. Skipping. Install with 'pip install shap'.")
        return

    try:
        sample_size = min(2000, len(X))
        if sample_size == 0: return
        shap_X = X.sample(n=sample_size, random_state=random_state)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(shap_X)
        plt.figure()
        shap.summary_plot(values, shap_X, plot_type="bar", show=False)
        plt.savefig(os.path.join(outputs_dir, "shap_summary_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure()
        shap.summary_plot(values, shap_X, show=False)
        plt.savefig(os.path.join(outputs_dir, "shap_summary_dot.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logging.warning("SHAP analysis skipped due to an error: %s", exc)
