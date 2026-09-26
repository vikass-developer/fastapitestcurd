from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response, status

from datautils import build_features

from ..dependencies import NOT_FOUND, PassengerRepoDep, PositiveId
from ..models import Passenger
from ..repositories import PassengerFilter

router = APIRouter(prefix="/passengers", tags=["passengers"])


@router.get("", response_model=list[Passenger], summary="Search passengers")
async def search_passengers(
    repo: PassengerRepoDep,
    pclass: Annotated[int | None, Query(ge=1, le=3, description="Ticket class")] = None,
    sex: Literal["male", "female"] | None = None,
    survived: bool | None = None,
    min_age: Annotated[float | None, Query(ge=0)] = None,
    max_age: Annotated[float | None, Query(ge=0)] = None,
    name: Annotated[
        str | None, Query(min_length=2, max_length=50, description="Name contains (any case)")
    ] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "min_age is greater than max_age"
        )
    filters = PassengerFilter(pclass, sex, survived, min_age, max_age, name)
    return await repo.search(filters, skip=skip, limit=limit)


# Declared before /{passenger_id} so "features" is not parsed as an id.
@router.get(
    "/features",
    summary="Feature matrix ready for model training",
    description=(
        "Numeric features built from every passenger: one-hot `pclass_*` and `embarked_*`, "
        "`is_female`, `family_size` and `is_alone`, with `survived` as the last column "
        "(the target). Use `?format=csv` to download a CSV file."
    ),
    responses={200: {"content": {"application/json": {}, "text/csv": {}}}},
)
async def features(repo: PassengerRepoDep, format: Literal["json", "csv"] = "json") -> Response:
    df = build_features(pd.DataFrame(await repo.all()))
    if format == "csv":
        return Response(
            df.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="titanic_features.csv"'},
        )
    return Response(df.to_json(orient="records"), media_type="application/json")


@router.get(
    "/{passenger_id}", response_model=Passenger, summary="Get a passenger", responses=NOT_FOUND
)
async def get_passenger(passenger_id: PositiveId, repo: PassengerRepoDep) -> dict:
    return await repo.get(passenger_id)
