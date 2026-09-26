"""FastAPI application exposing CRUD endpoints for items.

Run with:  uvicorn app.main:app --reload --app-dir src
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .models import Item, ItemCreate, ItemUpdate
from .repository import ItemNotFoundError, ItemRepository


def create_app() -> FastAPI:
    app = FastAPI(title="Items CRUD API", version="0.1.0")
    app.state.repo = ItemRepository()

    @app.exception_handler(ItemNotFoundError)
    async def item_not_found(_: Request, exc: ItemNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Item {exc.item_id} not found"},
        )

    def get_repo(request: Request) -> ItemRepository:
        return request.app.state.repo

    Repo = Annotated[ItemRepository, Depends(get_repo)]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/items", response_model=list[Item])
    async def list_items(
        repo: Repo,
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        tag: str | None = None,
    ) -> list[Item]:
        return await repo.list(skip=skip, limit=limit, tag=tag)

    @app.get("/items/{item_id}", response_model=Item)
    async def get_item(item_id: int, repo: Repo) -> Item:
        return await repo.get(item_id)

    @app.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
    async def create_item(payload: ItemCreate, repo: Repo) -> Item:
        return await repo.create(payload)

    @app.patch("/items/{item_id}", response_model=Item)
    async def update_item(item_id: int, payload: ItemUpdate, repo: Repo) -> Item:
        if not payload.model_fields_set:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")
        return await repo.update(item_id, payload)

    @app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_item(item_id: int, repo: Repo) -> None:
        await repo.delete(item_id)

    return app


app = create_app()
