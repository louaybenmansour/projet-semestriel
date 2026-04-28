"""Production-grade training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    FAIL_THRESHOLD,
    MODELS_DIR,
    NUMERIC_COLUMNS,
    ORDINAL_ORDER,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RAW_FEATURE_COLUMNS,
    TARGET_SCORE,
    ensure_directories,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


@dataclass
class TrainingArtifacts:
    """Container for trained model assets."""

    model_bundle: dict[str, Any]
    metrics: pd.DataFrame
    best_model_name: str
    feature_importance: pd.DataFrame


def load_base_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load CSV and keep only source columns + target score."""
    csv_path = path or RAW_DATA_PATH
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Fallback for malformed lines in external CSV exports.
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    keep = [c for c in RAW_FEATURE_COLUMNS + [TARGET_SCORE] if c in df.columns]
    missing = set(RAW_FEATURE_COLUMNS + [TARGET_SCORE]) - set(keep)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df[keep].copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional meaningful features used by models and UI."""
    out = df.copy()
    hs = pd.to_numeric(out["Hours_Studied"], errors="coerce").fillna(0.0)
    att = pd.to_numeric(out["Attendance"], errors="coerce").fillna(0.0)
    sleep = pd.to_numeric(out["Sleep_Hours"], errors="coerce").fillna(7.0)
    sleep_safe = np.maximum(sleep, 0.5)
    out["Academic_Stress_Index"] = (
        hs * (100.0 - att)
    ) / sleep_safe
    out["Digital_Access_Score"] = (
        out["Internet_Access"].astype(str).map({"No": 0, "Yes": 3}).fillna(0).astype(float)
        + out["Access_to_Resources"].astype(str).map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(float)
    )
    if TARGET_SCORE in out.columns:
        out["Risk"] = (out[TARGET_SCORE] < FAIL_THRESHOLD).astype(int)
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning with conservative imputations."""
    out = df.copy().drop_duplicates()
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in out.columns] + [TARGET_SCORE]
    numeric_cols = list(dict.fromkeys(numeric_cols))
    cat_cols = [c for c in out.columns if c not in numeric_cols]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    for col in cat_cols:
        out[col] = out[col].astype("string")
        mode = out[col].mode(dropna=True)
        out[col] = out[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    out = out.dropna(subset=[TARGET_SCORE]).reset_index(drop=True)
    # Remove clearly corrupted rows outside valid target range.
    out = out[(out[TARGET_SCORE] >= 0) & (out[TARGET_SCORE] <= 100)].reset_index(drop=True)
    return out


def _feature_schema(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    feature_cols = [c for c in df.columns if c not in [TARGET_SCORE, "Risk"]]
    ordinal_cols = [c for c in ORDINAL_ORDER if c in feature_cols]
    ordinal_categories = [ORDINAL_ORDER[c] for c in ordinal_cols]
    categorical_cols = [
        c
        for c in feature_cols
        if c not in ordinal_cols and str(df[c].dtype).startswith(("object", "string"))
    ]
    numeric_cols = [c for c in feature_cols if c not in ordinal_cols + categorical_cols]
    return feature_cols, numeric_cols, categorical_cols, ordinal_categories


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, dict[str, Any]]:
    """Build robust preprocessing graph for mixed data."""
    feature_cols, numeric_cols, categorical_cols, ordinal_categories = _feature_schema(df)
    ordinal_cols = [c for c in ORDINAL_ORDER if c in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "ordinal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=ordinal_categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ordinal_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    metadata = {
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "ordinal_columns": ordinal_cols,
    }
    return preprocessor, metadata


def _candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=8,
            min_samples_leaf=6,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        )
    return models


def train_and_select_best(df: pd.DataFrame) -> TrainingArtifacts:
    """Train multiple regressors, compare metrics, and return best model bundle."""
    preprocessor, schema = build_preprocessor(df)
    feature_cols = schema["feature_columns"]
    X = df[feature_cols]
    y = df[TARGET_SCORE].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    metrics_rows: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}
    for name, model in _candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics_rows.append(
            {
                "model": name,
                "MAE": float(mean_absolute_error(y_test, preds)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                "R2": float(r2_score(y_test, preds)),
            }
        )
        trained_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    importance_df = _compute_feature_importance(best_pipeline)
    bundle = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "schema": schema,
        "threshold_fail": FAIL_THRESHOLD,
    }
    return TrainingArtifacts(
        model_bundle=bundle,
        metrics=metrics_df,
        best_model_name=best_model_name,
        feature_importance=importance_df,
    )


def _compute_feature_importance(trained_pipeline: Pipeline) -> pd.DataFrame:
    """Extract top feature importances from tree model when available."""
    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        imp = pd.DataFrame({"feature": names, "importance": values})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame({"feature": names, "importance": np.nan})


def save_training_artifacts(artifacts: TrainingArtifacts) -> dict[str, Path]:
    ensure_directories()
    model_path = MODELS_DIR / "model.pkl"
    legacy_model_path = MODELS_DIR / "student_score_model.joblib"
    metrics_path = MODELS_DIR / "model_metrics.csv"
    feat_imp_path = MODELS_DIR / "feature_importance.csv"
    joblib.dump(artifacts.model_bundle, model_path)
    # Backward-compatible copy for older scripts.
    joblib.dump(artifacts.model_bundle, legacy_model_path)
    artifacts.metrics.to_csv(metrics_path, index=False)
    artifacts.feature_importance.to_csv(feat_imp_path, index=False)
    return {
        "model": model_path,
        "legacy_model": legacy_model_path,
        "metrics": metrics_path,
        "feature_importance": feat_imp_path,
    }


def load_model_bundle(model_path: Path | None = None) -> dict[str, Any]:
    path = model_path or MODELS_DIR / "model.pkl"
    if not path.exists():
        legacy = MODELS_DIR / "student_score_model.joblib"
        if legacy.exists():
            path = legacy
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def predict_score(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Predict score and classify pass/fail from JSON-like payload."""
    schema = bundle["schema"]
    threshold = bundle.get("threshold_fail", FAIL_THRESHOLD)
    base_cols = schema["feature_columns"]
    row = {col: payload.get(col) for col in base_cols}
    frame = pd.DataFrame([row])
    for c in NUMERIC_COLUMNS:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = add_engineered_features(frame)
    score = float(bundle["pipeline"].predict(frame[base_cols])[0])
    score = max(0.0, min(100.0, score))
    risk = int(score < threshold)
    return {
        "predicted_exam_score": round(score, 2),
        "predicted_risk": risk,
        "predicted_label": "Fail" if risk == 1 else "Pass",
    }
