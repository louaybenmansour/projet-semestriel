#!/usr/bin/env python3
"""Point d'entrée: exécuter depuis la racine du projet (recommandé)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
