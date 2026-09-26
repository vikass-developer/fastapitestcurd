"""FastAPI dependencies that hand routes the repositories stored on app.state."""

from typing import Annotated

from fastapi import Depends, Path, Request, status

from .models import ErrorResponse
from .repositories import ItemStore, PassengerRepository


def get_item_store(request: Request) -> ItemStore:
    return request.app.state.item_store


def get_passenger_repo(request: Request) -> PassengerRepository:
    return request.app.state.passenger_repo


ItemStoreDep = Annotated[ItemStore, Depends(get_item_store)]
PassengerRepoDep = Annotated[PassengerRepository, Depends(get_passenger_repo)]

PositiveId = Annotated[int, Path(ge=1, examples=[1])]

NOT_FOUND = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Not found"}}
