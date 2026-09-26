"""Shared fixtures.

SQL tests use a throwaway database (week1_ai_test) that is created, loaded and dropped per
session. They are skipped automatically when SQL Server cannot be reached.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pyodbc
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from datautils import read_table
from db import Database, DbSettings
from db.init_db import apply_schema, create_database, drop_database, load_passengers

CLEAN_CSV = Path(__file__).parents[1] / "data" / "processed" / "titanic_clean.csv"


@pytest.fixture(scope="session")
def sql_settings() -> Iterator[Settings]:
    db_settings = replace(DbSettings.from_env(), database="week1_ai_test")
    try:
        create_database(db_settings)
    except pyodbc.Error as exc:
        pytest.skip(f"SQL Server not available: {exc}")
    try:
        db = Database(db_settings.conn_str())
        apply_schema(db)
        load_passengers(db, read_table(CLEAN_CSV))
        yield Settings(storage="sql", db=db_settings)
    finally:
        drop_database(db_settings)


@pytest.fixture
def sql_client(sql_settings: Settings) -> TestClient:
    Database(sql_settings.db.conn_str()).execute("DELETE FROM dbo.items")  # tags cascade
    return TestClient(create_app(sql_settings))


@pytest.fixture(params=["memory", "sql"])
def client(request: pytest.FixtureRequest) -> TestClient:
    """Runs each items test once per storage backend."""
    if request.param == "memory":
        return TestClient(create_app(Settings(storage="memory")))
    return request.getfixturevalue("sql_client")
