import logging
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Explicit allowlist prevents counts, diagnostics, and target-derived labels leaking in.
    if "LST_Celsius" not in df.columns or "Land_Cover_Type" not in df.columns:
        raise ValueError("ML requires LST_Celsius and Land_Cover_Type columns")
    predictors = [
        "NDVI",
        "NDBI",
        "Elevation",
        "Slope",
        "Aspect_sin",
        "Aspect_cos",
        "Land_Cover_Type",
    ]
    columns = [c for c in predictors if c in df.columns] + ["LST_Celsius"]
    df_ml = pd.get_dummies(
        df[columns], columns=["Land_Cover_Type"], prefix="LC", dtype=float
    )

    X = df_ml.drop(columns=["LST_Celsius"])
    y = df_ml["LST_Celsius"]

    if (
        X.empty
        or y.empty
        or not np.isfinite(X.to_numpy()).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Features and target must be nonempty and finite")

    logging.info(
        "Features prepared for ML. Shape: %s. Features: %s", X.shape, list(X.columns)
    )
    return X, y


def spatial_groups(df: pd.DataFrame, block_size_m: int = 5000) -> np.ndarray:
    """Fixed equal-area spatial blocks keep repeated years at a location together."""
    if block_size_m <= 0:
        raise ValueError("Block size must be positive")
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform(
        df.longitude.to_numpy(), df.latitude.to_numpy()
    )
    cells = np.column_stack(
        [np.floor(np.asarray(x) / block_size_m), np.floor(np.asarray(y) / block_size_m)]
    )
    return np.unique(cells, axis=0, return_inverse=True)[1]


def spatial_holdout(df: pd.DataFrame, random_state: int, block_size_m: int = 5000):
    groups = spatial_groups(df, block_size_m)
    if len(df) < 10 or len(np.unique(groups)) < 3:
        raise ValueError(
            "Spatial validation needs >=10 samples across >=3 blocks; adjust ROI or block size"
        )
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(
            df, groups=groups
        )
    )
    if min(len(train), len(test)) < 2:
        raise ValueError("Too few samples in spatial training/test split")
    return train, test, groups


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    perform_tuning: bool,
    tuning_iterations: int,
    default_params: Dict,
    groups=None,
) -> RandomForestRegressor:
    if perform_tuning:
        logging.info("Hyperparameter tuning (%d iterations)...", tuning_iterations)
        if groups is None or len(np.unique(groups)) < 2:
            raise ValueError(
                "Tuning requires at least two spatial groups in training data"
            )
        rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_features": ["sqrt", "log2", 1.0],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=tuning_iterations,
            cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))),
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        search.fit(X, y, groups=groups)
        logging.info("Best params: %s", search.best_params_)
        return search.best_estimator_
    else:
        logging.info("Training RF with default parameters.")
        model = RandomForestRegressor(
            random_state=random_state, n_jobs=-1, **default_params
        )
        model.fit(X, y)
        return model


def evaluate_and_plot(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outputs_dir: str,
) -> Dict[str, float]:
    """Evaluate the fitted model on the held-out blocks; never fit on these rows."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=25, color="mediumblue")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0, 0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
    axes[0, 0].set_title(
        f"Model Predictions vs. Actual Temperatures (R²={r2:.3f})", fontsize=14
    )
    axes[0, 0].set_xlabel("Actual Temperature (°C)")
    axes[0, 0].set_ylabel("Predicted Temperature (°C)")

    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=25, color="darkgreen")
    axes[0, 1].axhline(0, color="r", linestyle="--")
    axes[0, 1].set_title("Prediction Errors (Residuals)", fontsize=14)
    axes[0, 1].set_xlabel("Predicted Temperature (°C)")
    axes[0, 1].set_ylabel("Prediction Error (°C)")

    importances = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(
        10
    )
    importances.plot(
        kind="barh", ax=axes[1, 0], color=sns.color_palette("viridis", len(importances))
    )
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
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "test_samples": len(y_test),
    }


def shap_analysis(
    model: RandomForestRegressor, X: pd.DataFrame, outputs_dir: str, random_state: int
) -> None:
    try:
        import shap
    except ImportError:
        logging.warning(
            "SHAP library not installed. Skipping. Install with 'pip install shap'."
        )
        return

    try:
        sample_size = min(2000, len(X))
        if sample_size == 0:
            return
        shap_X = X.sample(n=sample_size, random_state=random_state)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(shap_X)
        plt.figure()
        shap.summary_plot(values, shap_X, plot_type="bar", show=False)
        plt.savefig(
            os.path.join(outputs_dir, "shap_summary_bar.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        plt.figure()
        shap.summary_plot(values, shap_X, show=False)
        plt.savefig(
            os.path.join(outputs_dir, "shap_summary_dot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:
        logging.warning("SHAP analysis skipped due to an error: %s", exc)
