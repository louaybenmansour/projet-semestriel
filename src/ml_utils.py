"""Utilitaires d'affichage et sélection de variables (Random Forest, corrélations)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import OUTPUTS_DIR, RANDOM_STATE


def print_section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_insights(lines: list[str]) -> None:
    print("\nKey insights:")
    for line in lines:
        print(f"  - {line}")


def feature_selection(
    X: pd.DataFrame, y: pd.Series, top_k: int = 12, *, generate_figures: bool = True
) -> tuple[pd.Series, pd.Series]:
    print_section("5. FEATURE SELECTION")

    corr_abs = X.corrwith(y).abs()
    corr_with_target = corr_abs.sort_values(ascending=False)
    print("\nTop correlations with Risk (absolute value):")
    print(corr_with_target.head(15))

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nRandom Forest feature importances (top):")
    print(imp.head(15))

    r_corr = corr_abs.rank(ascending=False)
    r_imp = imp.rank(ascending=False)
    common = r_corr.index.intersection(r_imp.index)
    score = 0.5 * r_corr[common] + 0.5 * r_imp[common]
    selected = score.nsmallest(top_k).index.tolist()
    print(
        f"\nSelected top {top_k} features (average rank of |corr(Risk)| and RF importance; lower is better):"
    )
    print(selected)

    if generate_figures:
        fig, ax = plt.subplots(figsize=(10, 6))
        imp.head(15).iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title("Random Forest feature importance (top 15)")
        plt.tight_layout()
        p = OUTPUTS_DIR / "06_feature_importance_rf.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")
    else:
        print("(Skipped figures: 06_feature_importance_rf.png not written)")

    print_insights(
        [
            "Correlation highlights linear/monotone associations with failure risk in the prepared matrix.",
            "Random Forest importance captures non-linear interactions and threshold effects.",
            "A combined ranking reduces reliance on any single criterion and stabilizes feature picks for reporting.",
        ]
    )
    return imp, corr_abs


def presentation_summary(
    df_clean: pd.DataFrame,
    corr_abs: pd.Series,
    imp: pd.Series,
    *,
    generate_figures: bool = True,
) -> None:
    title = (
        "6. VISUALIZATIONS + 7. OUTPUT / PRESENTATION SUMMARY"
        if generate_figures
        else "7. OUTPUT / PRESENTATION SUMMARY (figures skipped)"
    )
    print_section(title)

    corr_top = corr_abs.sort_values(ascending=False)
    imp_top = imp.sort_values(ascending=False)
    drivers = list(dict.fromkeys(list(corr_top.head(8).index) + list(imp_top.head(8).index)))[:10]
    print("\nKey factors influencing student performance / failure risk (evidence-guided shortlist):")
    for d in drivers:
        print(f"  - {d}")

    print(
        "\nBusiness interpretation:\n"
        "  Schools can use risk indicators to prioritize counseling, attendance interventions, and\n"
        "  academic support. Factors with strong association/importance suggest where operational\n"
        "  investments (tutoring, parental engagement programs, infrastructure) may yield the largest\n"
        "  reduction in failure risk, especially when changes are measurable (attendance, sessions,\n"
        "  resource access) rather than purely demographic."
    )
