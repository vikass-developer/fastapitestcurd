"""The five analytics queries, one endpoint each."""

from typing import Annotated

from fastapi import APIRouter, Query

from ..dependencies import PassengerRepoDep
from ..models import (
    AgeGroupSurvival,
    ClassSexSurvival,
    FamilySizeSurvival,
    PortSummary,
    RankedPassenger,
)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get(
    "/survival-by-class-sex",
    response_model=list[ClassSexSurvival],
    summary="1. Survival rate by ticket class and sex (GROUP BY)",
)
async def survival_by_class_sex(repo: PassengerRepoDep) -> list[dict]:
    return await repo.survival_by_class_and_sex()


@router.get(
    "/survival-by-age-group",
    response_model=list[AgeGroupSurvival],
    summary="2. Survival rate by age group (CTE + CASE)",
)
async def survival_by_age_group(repo: PassengerRepoDep) -> list[dict]:
    return await repo.survival_by_age_group()


@router.get(
    "/oldest-per-class",
    response_model=list[RankedPassenger],
    summary="3. Oldest N passengers in each class (ROW_NUMBER window)",
)
async def oldest_per_class(
    repo: PassengerRepoDep, top: Annotated[int, Query(ge=1, le=20)] = 3
) -> list[dict]:
    return await repo.oldest_per_class(top)


@router.get(
    "/survival-by-family-size",
    response_model=list[FamilySizeSurvival],
    summary="4. Survival rate by family size (GROUP BY expression + HAVING)",
)
async def survival_by_family_size(
    repo: PassengerRepoDep,
    min_passengers: Annotated[int, Query(ge=1, description="Hide smaller groups")] = 5,
) -> list[dict]:
    return await repo.survival_by_family_size(min_passengers)


@router.get(
    "/embarked",
    response_model=list[PortSummary],
    summary="5. Summary by port of embarkation (JOIN + window SUM)",
)
async def embarked(repo: PassengerRepoDep) -> list[dict]:
    return await repo.embarked_summary()
