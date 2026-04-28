"""Encodage ordinal / one-hot (séparé pour éviter les imports circulaires)."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.config import NOMINAL_COLUMNS, ORDINAL_ORDER


def label_encode_ordinals(df: pd.DataFrame) -> tuple[pd.DataFrame, OrdinalEncoder, list[str]]:
    df_enc = df.copy()
    ordinal_cols = [c for c in ORDINAL_ORDER if c in df_enc.columns]
    categories = [ORDINAL_ORDER[c] for c in ordinal_cols]
    enc = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df_enc[ordinal_cols] = enc.fit_transform(df_enc[ordinal_cols].astype(str))
    return df_enc, enc, ordinal_cols


def one_hot_nominals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    nominals = [c for c in NOMINAL_COLUMNS if c in df.columns]
    return pd.get_dummies(df, columns=nominals, drop_first=True), nominals
