"""Graphiques EDA (matplotlib / seaborn), sauvegardés dans outputs/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    FAIL_THRESHOLD,
    NUMERIC_COLUMNS,
    ORDINAL_ORDER,
    OUTPUTS_DIR,
    TARGET_SCORE,
)
from src.encoding import label_encode_ordinals
from src.ml_utils import print_insights, print_section


def plot_cleaning_boxplots(
    df_clean: pd.DataFrame, num_cols: list[str], *, generate_figures: bool = True
) -> None:
    if not generate_figures:
        print("(Skipped figures: 01_boxplots_numeric_outliers.png not written)")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    plot_cols = num_cols + [TARGET_SCORE]
    for ax, col in zip(axes, plot_cols):
        sns.boxplot(data=df_clean, y=col, ax=ax, color="#4C72B0")
        ax.set_title(f"Boxplot: {col}")
    for j in range(len(plot_cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    fig_path = OUTPUTS_DIR / "01_boxplots_numeric_outliers.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path}")


def exploratory_data_analysis(df_clean: pd.DataFrame, *, generate_figures: bool = True) -> None:
    print_section("3. EXPLORATORY DATA ANALYSIS (EDA)")

    sns.set_theme(style="whitegrid", context="talk")
    df = df_clean.copy()
    df["Risk"] = (df[TARGET_SCORE] < FAIL_THRESHOLD).astype(int)

    print("\n--- A. Univariate analysis ---")

    if generate_figures:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        sns.histplot(df[TARGET_SCORE], kde=True, ax=axes[0, 0], color="#55A868")
        axes[0, 0].set_title("Distribution of Exam_Score")
        sns.boxplot(df[TARGET_SCORE], ax=axes[0, 1], orient="h", color="#C44E52")
        axes[0, 1].set_title("Boxplot of Exam_Score")

        sns.histplot(df["Hours_Studied"], kde=True, ax=axes[1, 0], color="#4C72B0")
        axes[1, 0].set_title("Distribution of Hours_Studied")
        sns.histplot(df["Attendance"], kde=True, ax=axes[1, 1], color="#8172B3")
        axes[1, 1].set_title("Distribution of Attendance")
        plt.tight_layout()
        p = OUTPUTS_DIR / "02_univariate_hist_exam_hours_attendance.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

        fig, ax = plt.subplots(figsize=(6, 4))
        vc = df["Risk"].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=["#55A868", "#C44E52"])
        ax.set_xlabel("Risk (0=pass, 1=fail)")
        ax.set_ylabel("Count")
        ax.set_title("Bar chart: Risk (failure) class frequency")
        plt.tight_layout()
        p_risk = OUTPUTS_DIR / "02b_bar_risk_distribution.png"
        plt.savefig(p_risk, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p_risk}")
    else:
        print("(Skipped figures: 02_*.png and 02b_*.png not written)")
        print("Risk class counts:", df["Risk"].value_counts().sort_index().to_dict())

    uni_insights = [
        "Exam_Score distribution shows where the cohort sits relative to the failure threshold (60).",
        "Hours_Studied and Attendance skew patterns highlight typical behavior ranges and long tails.",
    ]
    if generate_figures:
        uni_insights.append(
            "The Risk bar chart highlights class imbalance; modeling may need balancing or appropriate metrics."
        )
    else:
        uni_insights.append(
            "Class counts printed above show imbalance when figures are disabled; use metrics suited to rare events."
        )
    print_insights(uni_insights)

    print("\n--- B. Bivariate analysis ---")

    if generate_figures:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.scatterplot(data=df, x="Hours_Studied", y=TARGET_SCORE, hue="Risk", alpha=0.35, ax=axes[0])
        axes[0].axhline(FAIL_THRESHOLD, color="black", ls="--", lw=1)
        axes[0].set_title("Exam_Score vs Hours_Studied (colored by Risk)")

        sns.scatterplot(data=df, x="Attendance", y=TARGET_SCORE, hue="Risk", alpha=0.35, ax=axes[1])
        axes[1].axhline(FAIL_THRESHOLD, color="black", ls="--", lw=1)
        axes[1].set_title("Exam_Score vs Attendance (colored by Risk)")

        sns.boxplot(
            data=df,
            x="Parental_Education_Level",
            y=TARGET_SCORE,
            hue="Parental_Education_Level",
            ax=axes[2],
            palette="Set2",
            legend=False,
        )
        axes[2].axhline(FAIL_THRESHOLD, color="black", ls="--", lw=1)
        axes[2].set_title("Exam_Score by Parental_Education_Level")
        axes[2].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        p = OUTPUTS_DIR / "03_bivariate_score_relationships.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")
    else:
        print("(Skipped figures: 03_bivariate_score_relationships.png not written)")

    print_insights(
        [
            "Scatterplots suggest whether more study time and higher attendance associate with higher scores "
            "and fewer points below the failure line.",
            "Boxplots by parental education summarize central tendency and spread differences across groups.",
        ]
    )

    print("\n--- C. Multivariate analysis ---")

    df_ord, _, _ = label_encode_ordinals(df)
    heat_cols = [c for c in NUMERIC_COLUMNS + [TARGET_SCORE] if c in df_ord.columns]
    for oc in ORDINAL_ORDER:
        if oc in df_ord.columns:
            heat_cols.append(oc)
    heat_cols = list(dict.fromkeys(heat_cols))
    corr = df_ord[heat_cols].corr(numeric_only=True)

    if generate_figures:
        plt.figure(figsize=(11, 9))
        sns.heatmap(corr, annot=False, cmap="vlag", center=0, linewidths=0.5)
        plt.title("Correlation heatmap (numeric + ordinal-encoded factors)")
        p = OUTPUTS_DIR / "04_correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")
    else:
        print("(Skipped figures: 04_correlation_heatmap.png not written)")
        ctop = corr[TARGET_SCORE].drop(labels=[TARGET_SCORE], errors="ignore").abs().sort_values(ascending=False).head(8)
        print("Top |corr| with Exam_Score (console preview):\n", ctop.to_string())

    print_insights(
        [
            "The heatmap highlights clusters of co-moving variables (e.g., attendance, motivation proxies).",
            "Strong correlation with Exam_Score indicates candidate drivers for risk modeling.",
        ]
    )


def plot_engineered_features(df: pd.DataFrame, *, generate_figures: bool = True) -> None:
    if not generate_figures:
        print("(Skipped figures: 05_engineered_features.png not written)")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Academic_Stress_Index"], kde=True, ax=ax[0], color="#DD8452")
    ax[0].set_title("Academic_Stress_Index distribution")
    dvc = df["Digital_Access_Score"].value_counts().sort_index()
    ax[1].bar(dvc.index.astype(str), dvc.values, color="#937860")
    ax[1].set_title("Digital_Access_Score frequencies")
    plt.tight_layout()
    p = OUTPUTS_DIR / "05_engineered_features.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")
