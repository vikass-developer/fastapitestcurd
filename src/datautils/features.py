"""Turn cleaned passenger rows into a numeric feature matrix ready for model training."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per passenger: numeric and one-hot features, with the `survived` target last."""
    out = pd.DataFrame({"passenger_id": df["passenger_id"]})
    out["age"] = df["age"].astype(float)
    out["fare"] = df["fare"].astype(float)
    out["family_size"] = df["sib_sp"] + df["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)
    out["is_female"] = (df["sex"] == "female").astype(int)
    categorical = df[["pclass", "embarked"]].astype(str)
    out = pd.concat([out, pd.get_dummies(categorical, dtype=int)], axis=1)
    out["survived"] = df["survived"].astype(int)
    return out
