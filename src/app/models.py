"""Pydantic schemas for the Item resource."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

Name = Annotated[str, Field(min_length=1, max_length=100, description="Display name")]
Description = Annotated[str | None, Field(max_length=500, description="Optional long text")]
Price = Annotated[float, Field(gt=0, le=1_000_000, description="Unit price, must be > 0")]
Tags = Annotated[
    list[str], Field(max_length=10, description="Up to 10 labels, stored lowercase and unique")
]


def _normalize_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    cleaned = [tag.strip().lower() for tag in tags]
    if any(not tag for tag in cleaned):
        raise ValueError("tags must not be blank")
    return list(dict.fromkeys(cleaned))  # dedupe, keep order


class _Strict(BaseModel):
    # Trim whitespace on strings and reject unknown fields in request bodies.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class ItemCreate(_Strict):
    """Payload for creating an item."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Notebook",
                    "description": "A5 ruled, 80 pages",
                    "price": 3.99,
                    "tags": ["office", "paper"],
                }
            ]
        }
    )

    name: Name
    description: Description = None
    price: Price
    tags: Tags = Field(default_factory=list)

    _tags = field_validator("tags")(_normalize_tags)


class ItemUpdate(_Strict):
    """Partial update: send only the fields to change. `description` may be set to null."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"price": 2.49}]})

    name: Name | None = None
    description: Description = None
    price: Price | None = None
    tags: Tags | None = None

    _tags = field_validator("tags")(_normalize_tags)

    @field_validator("name", "price", "tags")
    @classmethod
    def _not_null(cls, value: object) -> object:
        # Omitting a field leaves it unchanged; explicitly sending null is an error.
        if value is None:
            raise ValueError("may be omitted but not null")
        return value


class Item(BaseModel):
    """An item as stored and returned by the API."""

    id: int = Field(description="Server-assigned identifier")
    name: str
    description: str | None = None
    price: float
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None, description="Null until first update")


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Item 42 not found"])
