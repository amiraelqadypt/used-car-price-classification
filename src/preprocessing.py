"""Cleaning, feature engineering, encoding and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Impute missing values, drop duplicates, coerce types, trim outliers."""
    out = df.copy()
    n_start = len(out)

    # 1. Missing values: median for numeric, mode for categorical.
    for col in out.select_dtypes(include=[np.number]).columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object"]).columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode()[0])

    # 2. Duplicates.
    n_before = len(out)
    out = out.drop_duplicates()
    n_dupes = n_before - len(out)

    # 3. Types.
    out["year"] = out["year"].astype(int)
    out["condition"] = pd.to_numeric(out["condition"], errors="coerce")
    out["odometer"] = pd.to_numeric(out["odometer"], errors="coerce")
    out = out.dropna(subset=["condition", "odometer", "mmr", "sellingprice"])

    # 4. Outliers on the target, 1st to 99th percentile.
    lo = out["sellingprice"].quantile(config.OUTLIER_LOWER_Q)
    hi = out["sellingprice"].quantile(config.OUTLIER_UPPER_Q)
    n_before = len(out)
    out = out[(out["sellingprice"] >= lo) & (out["sellingprice"] <= hi)]
    n_outliers = n_before - len(out)

    if verbose:
        print(f"  raw records          : {n_start:,}")
        print(f"  duplicates removed   : {n_dupes:,}")
        print(f"  outliers removed     : {n_outliers:,} "
              f"(outside ${lo:,.0f}-${hi:,.0f})")
        print(f"  records retained     : {len(out):,}")

    return out.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add car_age, price_diff and price_ratio.

    price_diff and price_ratio are derived from sellingprice and are therefore
    only meaningful where the transaction price is already known. They are used
    as model inputs in 'thesis' mode only; see config.MODES.
    """
    out = df.copy()
    out["car_age"] = config.REFERENCE_YEAR - out["year"]
    out["price_diff"] = out["sellingprice"] - out["mmr"]
    out["price_ratio"] = out["sellingprice"] / (out["mmr"] + 1)
    return out


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Integer-encode the categorical columns. Returns the frame and encoders."""
    out = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_FEATURES:
        enc = LabelEncoder()
        out[f"{col}_encoded"] = enc.fit_transform(out[col].astype(str))
        encoders[col] = enc
    return out, encoders


def build_target(df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Discretise sellingprice into equal-frequency price ranges.

    Returns the labelled series and the quartile boundaries.
    """
    labels, bins = pd.qcut(
        df["sellingprice"],
        q=config.N_PRICE_CLASSES,
        labels=config.PRICE_CLASS_LABELS,
        retbins=True,
    )
    return labels, bins


def feature_columns(include_price_derived: bool) -> list[str]:
    """The ordered list of classifier input columns for a given run mode."""
    numeric = list(config.BASE_NUMERIC_FEATURES)
    if include_price_derived:
        numeric += list(config.PRICE_DERIVED_FEATURES)
    encoded = [f"{c}_encoded" for c in config.CATEGORICAL_FEATURES]
    return numeric + encoded


def clustering_columns(include_price_derived: bool) -> list[str]:
    """The ordered list of columns supplied to K-Means.

    By default this excludes every feature derived from sellingprice, so the
    clustering step has no access to target information and the cluster label
    cannot carry it into the classifier.
    """
    numeric = list(config.BASE_NUMERIC_FEATURES)
    if include_price_derived:
        numeric += list(config.PRICE_DERIVED_FEATURES)
    encoded = [f"{c}_encoded" for c in config.CATEGORICAL_FEATURES]
    return numeric + encoded
