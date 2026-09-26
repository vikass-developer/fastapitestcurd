"""Create the database, apply schema.sql and load the cleaned Titanic CSV.

Usage:  init-db [--database NAME] [--csv PATH]
"""

from __future__ import annotations

import argparse
import re
from contextlib import closing
from pathlib import Path

import pandas as pd
import pyodbc

from datautils import read_table

from .config import DbSettings
from .connection import Database

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_CSV = Path("data/processed/titanic_clean.csv")
PASSENGER_COLUMNS = [
    "passenger_id", "survived", "pclass", "name", "sex", "age",
    "sib_sp", "parch", "ticket", "fare", "embarked",
]


def _safe_name(name: str) -> str:
    # CREATE/DROP DATABASE cannot take parameters, so only allow plain identifiers.
    if not re.fullmatch(r"\w{1,128}", name):
        raise ValueError(f"Invalid database name: {name!r}")
    return name


def _master(settings: DbSettings) -> closing[pyodbc.Connection]:
    # pyodbc's own context manager commits but does not close; closing() does.
    return closing(pyodbc.connect(settings.conn_str("master"), autocommit=True))


def create_database(settings: DbSettings) -> None:
    name = _safe_name(settings.database)
    with _master(settings) as conn:
        conn.execute(f"IF DB_ID(?) IS NULL CREATE DATABASE [{name}]", name)


def drop_database(settings: DbSettings) -> None:
    name = _safe_name(settings.database)
    with _master(settings) as conn:
        conn.execute(
            f"IF DB_ID(?) IS NOT NULL BEGIN "
            f"ALTER DATABASE [{name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; "
            f"DROP DATABASE [{name}]; END",
            name,
        )


def apply_schema(db: Database) -> None:
    batches = re.split(r"^\s*GO\s*$", SCHEMA_PATH.read_text(encoding="utf-8"), flags=re.M | re.I)
    with db.transaction() as cur:
        for batch in filter(str.strip, batches):
            cur.execute(batch)


def load_passengers(db: Database, df: pd.DataFrame) -> int:
    """Replace dbo.passengers with the rows of df. Returns the number of rows loaded."""
    rows = df[PASSENGER_COLUMNS].to_dict("split")["data"]  # native Python types
    placeholders = ", ".join("?" * len(PASSENGER_COLUMNS))
    with db.transaction() as cur:
        cur.execute("DELETE FROM dbo.passengers")
        cur.fast_executemany = True
        cur.executemany(
            f"INSERT INTO dbo.passengers ({', '.join(PASSENGER_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", help="Override MSSQL_DATABASE")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)

    settings = DbSettings.from_env()
    if args.database:
        settings = DbSettings(settings.server, args.database, settings.driver, settings.auth)
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found - run `clean-dataset` first.")

    create_database(settings)
    db = Database(settings.conn_str())
    apply_schema(db)
    count = load_passengers(db, read_table(args.csv))
    print(f"Database [{settings.database}] on {settings.server} ready: {count} passengers loaded.")


if __name__ == "__main__":
    main()
