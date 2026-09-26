"""FastAPI application exposing CRUD endpoints for items.

Run with:  uvicorn app.main:app --reload --app-dir src
Docs:      /docs (Swagger UI), /redoc (ReDoc), /openapi.json (raw schema)
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse

from .models import ErrorResponse, Item, ItemCreate, ItemUpdate
from .repository import ItemNotFoundError, ItemRepository

DESCRIPTION = """
CRUD API for **items**, backed by an async in-memory store.

* Request bodies are validated by Pydantic: unknown fields are rejected, strings are trimmed,
  and tags are stored lowercase without duplicates.
* Invalid input returns **422** with the field-level errors, and an unknown id returns **404**.
"""

NOT_FOUND = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Item not found"}}

ItemId = Annotated[int, Path(ge=1, description="Item id", examples=[1])]


def get_repo(request: Request) -> ItemRepository:
    return request.app.state.repo


Repo = Annotated[ItemRepository, Depends(get_repo)]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Items CRUD API",
        version="0.1.0",
        description=DESCRIPTION,
        openapi_tags=[
            {"name": "items", "description": "Create, read, update and delete items"},
            {"name": "system", "description": "Operational endpoints"},
        ],
    )
    app.state.repo = ItemRepository()

    @app.exception_handler(ItemNotFoundError)
    async def item_not_found(_: Request, exc: ItemNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Item {exc.item_id} not found"},
        )

    @app.get("/health", tags=["system"], summary="Liveness check")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/items", response_model=list[Item], tags=["items"], summary="List items")
    async def list_items(
        repo: Repo,
        skip: Annotated[int, Query(ge=0, description="Number of items to skip")] = 0,
        limit: Annotated[int, Query(ge=1, le=100, description="Max items to return")] = 20,
        tag: Annotated[str | None, Query(description="Only items with this tag")] = None,
    ) -> list[Item]:
        tag = tag.strip().lower() if tag else None
        return await repo.list(skip=skip, limit=limit, tag=tag)

    @app.get(
        "/items/{item_id}", response_model=Item, tags=["items"], summary="Get an item",
        responses=NOT_FOUND,
    )
    async def get_item(item_id: ItemId, repo: Repo) -> Item:
        return await repo.get(item_id)

    @app.post(
        "/items", response_model=Item, status_code=status.HTTP_201_CREATED, tags=["items"],
        summary="Create an item",
    )
    async def create_item(payload: ItemCreate, repo: Repo) -> Item:
        return await repo.create(payload)

    @app.patch(
        "/items/{item_id}", response_model=Item, tags=["items"], summary="Update an item",
        responses={
            **NOT_FOUND,
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Empty body"},
        },
    )
    async def update_item(item_id: ItemId, payload: ItemUpdate, repo: Repo) -> Item:
        if not payload.model_fields_set:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")
        return await repo.update(item_id, payload)

    @app.delete(
        "/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["items"],
        summary="Delete an item", responses=NOT_FOUND,
    )
    async def delete_item(item_id: ItemId, repo: Repo) -> None:
        await repo.delete(item_id)

    return app


app = create_app()
