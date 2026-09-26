"""CSV / JSON read and write helpers chosen by file extension."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a .csv, .json (array of records) or .jsonl file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.json_normalize(data)  # flattens nested objects into dotted columns
    raise ValueError(f"Unsupported file type: {suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame as .csv, .json (records) or .jsonl, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    elif suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return path
