"""Train and persist the best regression model for student score prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.config import OUTPUTS_DIR, ensure_directories
from src.production_ml import (
    add_engineered_features,
    clean_data,
    load_base_dataset,
    save_training_artifacts,
    train_and_select_best,
)


def run_eda(df: pd.DataFrame) -> None:
    """Save concise EDA artifacts for presentation and diagnostics."""
    ensure_directories()
    print("\n=== EDA Summary ===")
    print(df.describe(include="all").transpose().head(25))
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(20))

    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    corr_path = OUTPUTS_DIR / "modeling_correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Correlation Heatmap (Modeling)")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=150)
    plt.close()

    dist_path = OUTPUTS_DIR / "modeling_exam_score_distribution.png"
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Exam_Score"], kde=True, bins=30)
    plt.axvline(60, color="red", linestyle="--", label="Fail threshold")
    plt.legend()
    plt.title("Exam Score Distribution")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved EDA figures: {corr_path} and {dist_path}")


def main() -> None:
    ensure_directories()
    print("Loading dataset...")
    raw = load_base_dataset()
    cleaned = clean_data(raw)
    prepared = add_engineered_features(cleaned)

    run_eda(prepared)
    artifacts = train_and_select_best(prepared)
    saved = save_training_artifacts(artifacts)

    print("\n=== Model Evaluation ===")
    print(artifacts.metrics.to_string(index=False))
    print(f"\nBest model: {artifacts.best_model_name}")
    for key, path in saved.items():
        print(f"{key}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
