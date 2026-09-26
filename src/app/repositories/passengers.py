"""Read-only repository over dbo.passengers: search, lookup and the five analytics queries."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from db import Database, queries
from db.connection import Row

from .base import NotFoundError

_COLUMNS = "passenger_id, survived, pclass, name, sex, age, sib_sp, parch, ticket, fare, embarked"


def _escape_like(text: str) -> str:
    # Treat %, _ and [ in user input as literal characters in a LIKE pattern.
    return re.sub(r"([\\%_\[])", r"\\\1", text)


@dataclass(frozen=True)
class PassengerFilter:
    pclass: int | None = None
    sex: str | None = None
    survived: bool | None = None
    min_age: float | None = None
    max_age: float | None = None
    name: str | None = None

    def to_where(self) -> tuple[str, list[object]]:
        """Build a parameterized WHERE clause from the filters that are set."""
        clauses: list[str] = []
        params: list[object] = []
        for column, op, value in (
            ("pclass", "=", self.pclass),
            ("sex", "=", self.sex),
            ("survived", "=", self.survived),
            ("age", ">=", self.min_age),
            ("age", "<=", self.max_age),
        ):
            if value is not None:
                clauses.append(f"{column} {op} ?")
                params.append(value)
        if self.name:
            clauses.append("name LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(self.name)}%")
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


class PassengerRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _fetch(self, sql: str, *params: object) -> list[Row]:
        return await asyncio.to_thread(self._db.fetch_all, sql, *params)

    async def search(self, filters: PassengerFilter, *, skip: int, limit: int) -> list[Row]:
        where, params = filters.to_where()
        sql = (
            f"SELECT {_COLUMNS} FROM dbo.passengers {where} "
            "ORDER BY passenger_id OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
        )
        return await self._fetch(sql, *params, skip, limit)

    async def get(self, passenger_id: int) -> Row:
        rows = await self._fetch(
            f"SELECT {_COLUMNS} FROM dbo.passengers WHERE passenger_id = ?", passenger_id
        )
        if not rows:
            raise NotFoundError("Passenger", passenger_id)
        return rows[0]

    async def all(self) -> list[Row]:
        return await self._fetch(f"SELECT {_COLUMNS} FROM dbo.passengers ORDER BY passenger_id")

    # The five analytics queries (SQL in db/queries.py).

    async def survival_by_class_and_sex(self) -> list[Row]:
        return await self._fetch(queries.SURVIVAL_BY_CLASS_AND_SEX)

    async def survival_by_age_group(self) -> list[Row]:
        return await self._fetch(queries.SURVIVAL_BY_AGE_GROUP)

    async def oldest_per_class(self, top_n: int) -> list[Row]:
        return await self._fetch(queries.OLDEST_PER_CLASS, top_n)

    async def survival_by_family_size(self, min_passengers: int) -> list[Row]:
        return await self._fetch(queries.SURVIVAL_BY_FAMILY_SIZE, min_passengers)

    async def embarked_summary(self) -> list[Row]:
        return await self._fetch(queries.EMBARKED_SUMMARY)
