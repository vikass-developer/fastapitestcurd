"""Pydantic schemas for items, passengers and the analytics endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

Name = Annotated[str, Field(min_length=1, max_length=100, description="Display name")]
Description = Annotated[str | None, Field(max_length=500, description="Optional long text")]
# Rounded to cents so the in-memory and SQL (DECIMAL(12,2)) stores return the same value.
Price = Annotated[
    float,
    Field(gt=0, le=1_000_000, description="Unit price, must be > 0 (rounded to 2 decimals)"),
    AfterValidator(lambda value: round(value, 2)),
]
Tags = Annotated[
    list[str],
    Field(max_length=10, description="Up to 10 labels (max 50 chars), stored lowercase and unique"),
]


def _normalize_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    cleaned = [tag.strip().lower() for tag in tags]
    if any(not tag or len(tag) > 50 for tag in cleaned):
        raise ValueError("tags must be 1-50 characters")
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


class Passenger(BaseModel):
    """One row of the cleaned Titanic dataset."""

    passenger_id: int
    survived: bool
    pclass: int = Field(description="Ticket class: 1, 2 or 3")
    name: str
    sex: str
    age: float
    sib_sp: int = Field(description="Siblings or spouses aboard")
    parch: int = Field(description="Parents or children aboard")
    ticket: str
    fare: float
    embarked: str = Field(description="C = Cherbourg, Q = Queenstown, S = Southampton")


class _SurvivalStats(BaseModel):
    passengers: int
    survivors: int
    survival_rate_pct: float


class ClassSexSurvival(_SurvivalStats):
    pclass: int
    sex: str


class AgeGroupSurvival(_SurvivalStats):
    age_group: str


class FamilySizeSurvival(_SurvivalStats):
    family_size: int


class RankedPassenger(BaseModel):
    pclass: int
    age_rank: int
    passenger_id: int
    name: str
    age: float
    survived: bool


class PortSummary(BaseModel):
    embarked: str
    port_name: str
    passengers: int
    avg_fare: float
    survival_rate_pct: float
    share_pct: float = Field(description="Share of all passengers, in percent")
