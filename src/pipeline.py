"""Orchestration CRISP-DM: nettoyage, EDA, ingénierie, préparation, sélection, exports."""

from __future__ import annotations

from src.config import PROCESSED_DIR, RAW_DATA_PATH, ensure_directories
from src.eda_plotting import exploratory_data_analysis
from src.feature_engineering import (
    build_modeling_matrix,
    data_cleaning,
    load_raw_table,
    run_feature_engineering_section,
)
from src.ml_utils import feature_selection, presentation_summary, print_section


def run_full_pipeline(data_path=None, *, generate_figures: bool = True) -> None:
    """
    Ordre d'exécution technique: nettoyage, EDA, ingénierie, matrice X/y, sélection.
    Le rapport CRISP-DM peut présenter les sections dans un autre ordre si besoin.

    Args:
        data_path: chemin optionnel vers le CSV brut.
        generate_figures: si False, aucun PNG n'est écrit (calculs et CSV inchangés).
    """
    ensure_directories()
    print_section("CRISP-DM: Data Preparation & EDA pipeline")
    path = data_path or RAW_DATA_PATH
    print(f"Data file: {path}")
    print(f"generate_figures={generate_figures}")

    df_raw = load_raw_table(path)
    df_clean, _ = data_cleaning(df_raw, generate_figures=generate_figures)

    exploratory_data_analysis(df_clean, generate_figures=generate_figures)

    df_eng = run_feature_engineering_section(df_clean, generate_figures=generate_figures)

    X, y, _ = build_modeling_matrix(df_clean)
    imp, corr_abs = feature_selection(X, y, top_k=12, generate_figures=generate_figures)

    presentation_summary(df_clean, corr_abs, imp, generate_figures=generate_figures)

    out_x = PROCESSED_DIR / "prepared_X.csv"
    out_y = PROCESSED_DIR / "prepared_y.csv"
    out_eng = PROCESSED_DIR / "dataset_with_engineered_features.csv"
    X.to_csv(out_x, index=False)
    y.to_csv(out_y, index=False)
    df_eng.to_csv(out_eng, index=False)

    print_section("Artifacts saved")
    print(f"  {out_x}")
    print(f"  {out_y}")
    print(f"  {out_eng}")
