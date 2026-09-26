"""SQL Server item store (dbo.items + dbo.item_tags).

Blocking pyodbc calls run in a worker thread via asyncio.to_thread, so the event loop stays free.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC

import pyodbc

from db import Database
from db.connection import Row

from ..models import Item, ItemCreate, ItemUpdate
from .base import NotFoundError

# Tags come back as a JSON array from a correlated subquery, so one round trip returns the
# item and its tags in order.
_SELECT_ITEMS = """
SELECT i.id, i.name, i.description, i.price, i.created_at, i.updated_at,
       (SELECT t.tag FROM dbo.item_tags AS t
        WHERE t.item_id = i.id ORDER BY t.position FOR JSON PATH) AS tags_json
FROM dbo.items AS i
"""

_UPDATABLE_COLUMNS = {"name", "description", "price"}


def _to_item(row: Row) -> Item:
    tags = [entry["tag"] for entry in json.loads(row.pop("tags_json") or "[]")]
    # DATETIME2 has no time zone; the values are written with SYSUTCDATETIME().
    for key in ("created_at", "updated_at"):
        if row[key] is not None:
            row[key] = row[key].replace(tzinfo=UTC)
    return Item(**row, tags=tags)


def _insert_tags(cur: pyodbc.Cursor, item_id: int, tags: list[str]) -> None:
    if tags:
        cur.executemany(
            "INSERT INTO dbo.item_tags (item_id, position, tag) VALUES (?, ?, ?)",
            [(item_id, position, tag) for position, tag in enumerate(tags)],
        )


class SqlItemRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def list(self, *, skip: int = 0, limit: int = 100, tag: str | None = None) -> list[Item]:
        sql, params = _SELECT_ITEMS, []
        if tag is not None:
            sql += (
                " WHERE EXISTS (SELECT 1 FROM dbo.item_tags AS f"
                " WHERE f.item_id = i.id AND f.tag = ?)"
            )
            params.append(tag)
        sql += " ORDER BY i.id OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
        rows = await asyncio.to_thread(self._db.fetch_all, sql, *params, skip, limit)
        return [_to_item(row) for row in rows]

    async def get(self, item_id: int) -> Item:
        return await asyncio.to_thread(self._get, item_id)

    async def create(self, data: ItemCreate) -> Item:
        return await asyncio.to_thread(self._create, data)

    async def update(self, item_id: int, data: ItemUpdate) -> Item:
        return await asyncio.to_thread(self._update, item_id, data)

    async def delete(self, item_id: int) -> None:
        # item_tags rows go too, via ON DELETE CASCADE.
        deleted = await asyncio.to_thread(
            self._db.execute, "DELETE FROM dbo.items WHERE id = ?", item_id
        )
        if deleted == 0:
            raise NotFoundError("Item", item_id)

    def _get(self, item_id: int) -> Item:
        row = self._db.fetch_one(_SELECT_ITEMS + " WHERE i.id = ?", item_id)
        if row is None:
            raise NotFoundError("Item", item_id)
        return _to_item(row)

    def _create(self, data: ItemCreate) -> Item:
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT INTO dbo.items (name, description, price) OUTPUT INSERTED.id "
                "VALUES (?, ?, ?)",
                data.name, data.description, data.price,
            )
            item_id = cur.fetchval()
            _insert_tags(cur, item_id, data.tags)
        return self._get(item_id)

    def _update(self, item_id: int, data: ItemUpdate) -> Item:
        changes = data.model_dump(exclude_unset=True)
        tags = changes.pop("tags", None)
        assert changes.keys() <= _UPDATABLE_COLUMNS  # column names come from the model only
        assignments = [f"{column} = ?" for column in changes] + ["updated_at = SYSUTCDATETIME()"]
        with self._db.transaction() as cur:
            cur.execute(
                f"UPDATE dbo.items SET {', '.join(assignments)} WHERE id = ?",
                *changes.values(), item_id,
            )
            if cur.rowcount == 0:
                raise NotFoundError("Item", item_id)
            if tags is not None:
                cur.execute("DELETE FROM dbo.item_tags WHERE item_id = ?", item_id)
                _insert_tags(cur, item_id, tags)
        return self._get(item_id)
