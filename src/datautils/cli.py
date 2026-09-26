"""Clean the Titanic public dataset with the datautils pipeline.

Usage:  python -m datautils.cli [input] [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from .cleaning import (
    clean_strings,
    clip_outliers,
    fill_missing,
    missing_report,
    standardize_columns,
)
from .io import read_table, write_table

DEFAULT_INPUT = Path("data/raw/titanic.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    src = Path(args[0]) if args else DEFAULT_INPUT
    out_dir = Path(args[1]) if len(args) > 1 else DEFAULT_OUTPUT_DIR

    raw = read_table(src)
    print(f"Loaded {len(raw)} rows from {src}\n\nBefore cleaning:\n{missing_report(raw)}\n")

    cleaned = (
        raw.pipe(standardize_columns)
        .pipe(clean_strings)
        .drop_duplicates()
        .drop(columns=["cabin"])  # ~77% missing, not worth imputing
        .pipe(fill_missing, numeric="median")
        .pipe(clip_outliers, columns=["fare"], k=3.0)
    )

    print(f"After cleaning:\n{missing_report(cleaned)}\n")
    for name in ("titanic_clean.csv", "titanic_clean.json"):
        print(f"Wrote {write_table(cleaned, out_dir / name)}")


if __name__ == "__main__":
    main()
