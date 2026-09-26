"""Five reusable, pure data-cleaning functions.

Each function takes a DataFrame and returns a new one (the input is never
mutated), so they can be chained with DataFrame.pipe().
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

NumericStrategy = Literal["median", "mean", "zero"]

_NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """1. Convert column names to snake_case, e.g. 'Ticket No.' -> 'ticket_no'."""

    def to_snake(name: object) -> str:
        text = str(name).strip()
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)  # camelCase -> camel_Case
        text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
        return text.strip("_").lower()

    return df.rename(columns=to_snake)


def clean_strings(df: pd.DataFrame, *, lowercase: bool = False) -> pd.DataFrame:
    """2. Trim whitespace in text columns and turn null-like tokens ('', 'N/A', '?') into NaN."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        series = out[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        series = series.mask(series.str.lower().isin(_NULL_TOKENS))
        if lowercase:
            series = series.str.lower()
        out[col] = series
    return out


def fill_missing(
    df: pd.DataFrame,
    *,
    numeric: NumericStrategy = "median",
    categorical_fill: str | None = None,
    overrides: dict[str, object] | None = None,
) -> pd.DataFrame:
    """3. Fill NaNs: numeric columns by median/mean/zero, others by mode (or a constant).

    `overrides` sets an explicit fill value for specific columns.
    """
    out = df.copy()
    overrides = overrides or {}
    for col in out.columns:
        if col in overrides:
            out[col] = out[col].fillna(overrides[col])
        elif not out[col].isna().any():
            continue
        elif pd.api.types.is_numeric_dtype(out[col]):
            value = {"median": out[col].median(), "mean": out[col].mean(), "zero": 0}[numeric]
            out[col] = out[col].fillna(value)
        else:
            mode = out[col].mode(dropna=True)
            value = categorical_fill if categorical_fill is not None else (
                mode.iloc[0] if not mode.empty else None
            )
            if value is not None:
                out[col] = out[col].fillna(value)
    return out


def clip_outliers(
    df: pd.DataFrame, columns: Iterable[str] | None = None, *, k: float = 1.5
) -> pd.DataFrame:
    """4. Cap numeric values to the Tukey fences [Q1 - k*IQR, Q3 + k*IQR] using NumPy."""
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.select_dtypes("number").columns)
    for col in cols:
        values = out[col].to_numpy(dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        out[col] = np.clip(values, q1 - k * iqr, q3 + k * iqr)
    return out


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """5. Per-column profile: dtype, missing count/percent and unique values, worst first."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )
    return report.sort_values("missing", ascending=False)
