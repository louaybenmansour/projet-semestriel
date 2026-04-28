"""Chargement, nettoyage, ingénierie de variables, matrice X/y pour la modélisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FAIL_THRESHOLD,
    NUMERIC_COLUMNS,
    RAW_DATA_PATH,
    RAW_FEATURE_COLUMNS,
    TARGET_SCORE,
)
from src.eda_plotting import plot_cleaning_boxplots, plot_engineered_features
from src.encoding import label_encode_ordinals, one_hot_nominals
from src.ml_utils import print_insights, print_section


def load_raw_table(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path)
    extra = [c for c in df.columns if c not in RAW_FEATURE_COLUMNS + [TARGET_SCORE]]
    if extra:
        print(
            f"Note: dropping {len(extra)} precomputed/extra column(s) from file "
            f"to rebuild targets and engineered features consistently: {extra[:6]}{'...' if len(extra) > 6 else ''}"
        )
    use_cols = [c for c in RAW_FEATURE_COLUMNS + [TARGET_SCORE] if c in df.columns]
    missing_expected = set(RAW_FEATURE_COLUMNS + [TARGET_SCORE]) - set(use_cols)
    if missing_expected:
        raise ValueError(f"CSV missing expected columns: {sorted(missing_expected)}")
    return df[use_cols].copy()


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Academic_Stress_Index" in out.columns and "Digital_Access_Score" in out.columns:
        return out

    sleep_safe = np.maximum(out["Sleep_Hours"].astype(float), 0.5)
    out["Academic_Stress_Index"] = (
        out["Hours_Studied"].astype(float) * (100.0 - out["Attendance"].astype(float))
    ) / sleep_safe

    internet_map = {"Yes": 1, "No": 0}
    resource_map = {"Low": 0, "Medium": 1, "High": 2}
    inet = out["Internet_Access"].astype(str).map(internet_map).fillna(0).astype(int)
    res = out["Access_to_Resources"].astype(str).map(resource_map).fillna(1).astype(int)
    out["Digital_Access_Score"] = inet * 3 + res
    return out


def data_cleaning(df: pd.DataFrame, *, generate_figures: bool = True) -> tuple[pd.DataFrame, dict]:
    print_section("1. DATA CLEANING")
    report: dict = {}

    print(f"Shape before cleaning: {df.shape}")
    report["shape_before"] = df.shape

    na_per_col = df.isna().sum()
    na_total = int(na_per_col.sum())
    print("\nMissing values per column (top):")
    print(na_per_col[na_per_col > 0].sort_values(ascending=False).head(20))
    report["missing_total"] = na_total

    df_clean = df.copy()
    num_cols = [c for c in NUMERIC_COLUMNS if c in df_clean.columns]
    cat_cols = [c for c in df_clean.columns if c not in num_cols + [TARGET_SCORE]]

    for c in num_cols:
        if df_clean[c].isna().any():
            fill = df_clean[c].mean()
            df_clean[c] = df_clean[c].fillna(fill)
    for c in cat_cols:
        if df_clean[c].isna().any():
            mode = df_clean[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            df_clean[c] = df_clean[c].fillna(fill)

    dup_count = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"\nDuplicate rows removed: {dup_count}")
    report["duplicates_removed"] = dup_count

    for c in num_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    for c in cat_cols:
        df_clean[c] = df_clean[c].astype("string").astype("object")

    df_clean[TARGET_SCORE] = pd.to_numeric(df_clean[TARGET_SCORE], errors="coerce")
    bad_score = df_clean[TARGET_SCORE].isna().sum()
    if bad_score:
        df_clean = df_clean.dropna(subset=[TARGET_SCORE])
    print(f"\nInvalid Exam_Score rows dropped (if any): {int(bad_score)}")

    outlier_summary = {}
    for c in num_cols:
        q1 = df_clean[c].quantile(0.25)
        q3 = df_clean[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df_clean[c] < low) | (df_clean[c] > high)
        outlier_summary[c] = int(mask.sum())
    print("\nIQR-based outlier counts (for documentation; not auto-removed):")
    for k, v in outlier_summary.items():
        print(f"  {k}: {v}")

    plot_cleaning_boxplots(df_clean, num_cols, generate_figures=generate_figures)

    print(f"\nShape after cleaning: {df_clean.shape}")
    report["shape_after"] = df_clean.shape

    outlier_insight = (
        "Outliers were flagged with boxplots and the IQR rule; we retain them unless "
        "domain knowledge confirms data errors (extreme study hours or attendance can be real)."
        if generate_figures
        else "Outliers were quantified with the IQR rule (counts above); no boxplot PNG in this run."
    )
    print_insights(
        [
            "Missing values were imputed with the mean (numeric) or mode (categorical), "
            "which preserves the central tendency / majority category for ML readiness.",
            "Duplicates were removed to avoid inflated performance estimates.",
            outlier_insight,
            "Numeric columns were coerced to numeric types to avoid accidental string behavior.",
        ]
    )
    return df_clean, report


def build_modeling_matrix(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    print_section("2. DATA PREPARATION")

    df = add_engineered_columns(df_clean.copy())
    df["Risk"] = (df[TARGET_SCORE] < FAIL_THRESHOLD).astype(int)
    print("\nRisk distribution:")
    print(df["Risk"].value_counts(dropna=False))
    print(f"Failure rate (Exam_Score < {FAIL_THRESHOLD}): {df['Risk'].mean():.2%}")

    df_ord, _, ordinal_cols = label_encode_ordinals(df)
    df_oh, nominal_cols = one_hot_nominals(df_ord)

    feature_cols = [c for c in df_oh.columns if c not in (TARGET_SCORE, "Risk")]
    X = df_oh[feature_cols]
    y = df_oh["Risk"]

    numeric_in_x = [c for c in NUMERIC_COLUMNS if c in X.columns]
    passthrough = [c for c in X.columns if c not in numeric_in_x]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_in_x,
            ),
            ("passthrough", "passthrough", passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_scaled = pd.DataFrame(pre.fit_transform(X), columns=numeric_in_x + passthrough, index=X.index)

    print(f"\nPrepared feature matrix shape: {X_scaled.shape}")
    print(f"Ordinal label-encoded columns: {ordinal_cols}")
    print(f"One-hot source columns: {nominal_cols}")

    print_insights(
        [
            "Risk is defined as 1 when Exam_Score < 60, else 0; aligned with institutional pass/fail cutoffs.",
            "Ordinal variables use label encoding to preserve natural ordering (e.g., education levels).",
            "Nominal variables use one-hot encoding to avoid false ordering (e.g., Gender, School_Type).",
            "Continuous numeric predictors are standardized for distance-based interpretation and stable scaling; "
            "tree models are scale-invariant, but standardization supports mixed pipelines and heatmaps.",
            "Exam_Score is excluded from X when predicting Risk to prevent target leakage.",
        ]
    )
    return X_scaled, y, pre


def run_feature_engineering_section(df_clean: pd.DataFrame, *, generate_figures: bool = True) -> pd.DataFrame:
    print_section("4. FEATURE ENGINEERING")
    df = add_engineered_columns(df_clean.copy())
    df["Risk"] = (df[TARGET_SCORE] < FAIL_THRESHOLD).astype(int)
    print(df[["Risk", "Academic_Stress_Index", "Digital_Access_Score"]].describe().T)
    plot_engineered_features(df, generate_figures=generate_figures)
    print_insights(
        [
            "Academic_Stress_Index rises when study pressure is high relative to attendance and sleep; "
            "it proxies 'overload' scenarios that may precede poor outcomes.",
            "Digital_Access_Score combines connectivity and resource access into a compact support indicator.",
        ]
    )
    return df
