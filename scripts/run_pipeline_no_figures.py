#!/usr/bin/env python3
"""
Même pipeline que run_pipeline.py, sans écrire de fichiers graphiques (.png).

Utile pour: exécutions rapides, serveurs sans backend graphique, CI, ou
regénération des seuls CSV dans outputs/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(generate_figures=False)
