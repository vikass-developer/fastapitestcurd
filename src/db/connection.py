"""Thin pyodbc wrapper: one connection per unit of work, rows returned as dicts.

pyodbc is synchronous; async callers run these methods with asyncio.to_thread.
ODBC connection pooling (on by default in pyodbc) makes a connect per call cheap.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import pyodbc

Row = dict[str, Any]


def _to_python(value: object) -> object:
    # DECIMAL/NUMERIC columns and ROUND() results come back as Decimal; the API speaks float.
    return float(value) if isinstance(value, Decimal) else value


def rows_to_dicts(cursor: pyodbc.Cursor) -> list[Row]:
    columns = [col[0] for col in cursor.description]
    return [
        {col: _to_python(val) for col, val in zip(columns, row, strict=True)}
        for row in cursor.fetchall()
    ]


class Database:
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str

    @contextmanager
    def transaction(self) -> Iterator[pyodbc.Cursor]:
        """Yield a cursor; commit on success, roll back on any error."""
        conn = pyodbc.connect(self.conn_str, autocommit=False)
        try:
            yield conn.cursor()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def fetch_all(self, sql: str, *params: object) -> list[Row]:
        with self.transaction() as cur:
            cur.execute(sql, *params)
            return rows_to_dicts(cur)

    def fetch_one(self, sql: str, *params: object) -> Row | None:
        rows = self.fetch_all(sql, *params)
        return rows[0] if rows else None

    def execute(self, sql: str, *params: object) -> int:
        """Run a statement and return the affected row count."""
        with self.transaction() as cur:
            cur.execute(sql, *params)
            return cur.rowcount

    def ping(self) -> bool:
        try:
            return self.fetch_one("SELECT 1 AS ok") is not None
        except pyodbc.Error:
            return False
