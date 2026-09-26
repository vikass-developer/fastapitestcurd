"""Application settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from db import DbSettings

Storage = Literal["sql", "memory"]


@dataclass(frozen=True)
class Settings:
    """APP_STORAGE=sql (default) uses SQL Server; APP_STORAGE=memory needs no database."""

    storage: Storage = "sql"
    db: DbSettings = field(default_factory=DbSettings)

    @classmethod
    def from_env(cls) -> Settings:
        storage = os.getenv("APP_STORAGE", "sql").lower()
        if storage not in ("sql", "memory"):
            raise ValueError(f"APP_STORAGE must be 'sql' or 'memory', got {storage!r}")
        return cls(storage=storage, db=DbSettings.from_env())
