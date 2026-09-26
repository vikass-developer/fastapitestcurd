"""Repository contract shared by the in-memory and SQL Server item stores."""

from __future__ import annotations

from typing import Protocol

from ..models import Item, ItemCreate, ItemUpdate


class NotFoundError(LookupError):
    def __init__(self, resource: str, key: object) -> None:
        super().__init__(f"{resource} {key} not found")
        self.resource = resource
        self.key = key


class ItemStore(Protocol):
    """Anything with these async methods can back the /items API."""

    async def list(
        self, *, skip: int = 0, limit: int = 100, tag: str | None = None
    ) -> list[Item]: ...

    async def get(self, item_id: int) -> Item: ...

    async def create(self, data: ItemCreate) -> Item: ...

    async def update(self, item_id: int, data: ItemUpdate) -> Item: ...

    async def delete(self, item_id: int) -> None: ...
