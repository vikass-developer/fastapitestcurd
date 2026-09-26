from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from ..dependencies import NOT_FOUND, ItemStoreDep, PositiveId
from ..models import ErrorResponse, Item, ItemCreate, ItemUpdate

router = APIRouter(prefix="/items", tags=["items"])


@router.get("", response_model=list[Item], summary="List items")
async def list_items(
    store: ItemStoreDep,
    skip: Annotated[int, Query(ge=0, description="Number of items to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Max items to return")] = 20,
    tag: Annotated[str | None, Query(description="Only items with this tag")] = None,
) -> list[Item]:
    tag = tag.strip().lower() if tag else None
    return await store.list(skip=skip, limit=limit, tag=tag)


@router.get("/{item_id}", response_model=Item, summary="Get an item", responses=NOT_FOUND)
async def get_item(item_id: PositiveId, store: ItemStoreDep) -> Item:
    return await store.get(item_id)


@router.post(
    "", response_model=Item, status_code=status.HTTP_201_CREATED, summary="Create an item"
)
async def create_item(payload: ItemCreate, store: ItemStoreDep) -> Item:
    return await store.create(payload)


@router.patch(
    "/{item_id}",
    response_model=Item,
    summary="Update an item",
    responses={
        **NOT_FOUND,
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Empty body"},
    },
)
async def update_item(item_id: PositiveId, payload: ItemUpdate, store: ItemStoreDep) -> Item:
    if not payload.model_fields_set:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")
    return await store.update(item_id, payload)


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an item",
    responses=NOT_FOUND,
)
async def delete_item(item_id: PositiveId, store: ItemStoreDep) -> None:
    await store.delete(item_id)
