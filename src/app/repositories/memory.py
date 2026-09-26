"""In-memory async item store, used for tests and for running without a database.

An asyncio.Lock guards the dict so concurrent requests cannot interleave reads and writes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from itertools import count

from ..models import Item, ItemCreate, ItemUpdate
from .base import NotFoundError


class InMemoryItemRepository:
    def __init__(self) -> None:
        self._items: dict[int, Item] = {}
        self._ids = count(1)
        self._lock = asyncio.Lock()

    async def list(self, *, skip: int = 0, limit: int = 100, tag: str | None = None) -> list[Item]:
        async with self._lock:
            items = list(self._items.values())
        if tag is not None:
            items = [item for item in items if tag in item.tags]
        return items[skip : skip + limit]

    async def get(self, item_id: int) -> Item:
        async with self._lock:
            try:
                return self._items[item_id]
            except KeyError:
                raise NotFoundError("Item", item_id) from None

    async def create(self, data: ItemCreate) -> Item:
        async with self._lock:
            item = Item(id=next(self._ids), **data.model_dump())
            self._items[item.id] = item
            return item

    async def update(self, item_id: int, data: ItemUpdate) -> Item:
        async with self._lock:
            if item_id not in self._items:
                raise NotFoundError("Item", item_id)
            changes = data.model_dump(exclude_unset=True)
            updated = self._items[item_id].model_copy(
                update={**changes, "updated_at": datetime.now(UTC)}
            )
            self._items[item_id] = updated
            return updated

    async def delete(self, item_id: int) -> None:
        async with self._lock:
            if self._items.pop(item_id, None) is None:
                raise NotFoundError("Item", item_id)
