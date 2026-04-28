"""
Fichier historique : la logique a été déplacée vers le package `src/`.

Exécuter le pipeline depuis la racine du projet :
    python scripts/run_pipeline.py

Ou depuis Python :
    from src.pipeline import run_full_pipeline
    run_full_pipeline()
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
