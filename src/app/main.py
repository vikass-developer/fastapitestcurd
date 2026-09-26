"""FastAPI application: items CRUD plus Titanic passenger search and analytics.

Run with:  uvicorn app.main:app --reload --app-dir src
Docs:      /docs (Swagger UI), /redoc (ReDoc), /openapi.json (raw schema)
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from db import Database

from .config import Settings
from .repositories import (
    InMemoryItemRepository,
    NotFoundError,
    PassengerRepository,
    SqlItemRepository,
)
from .routers import items, passengers, stats, system

DESCRIPTION = """
A small backend with data ready for AI work, built on **FastAPI** and **SQL Server**.

* **items**: CRUD with Pydantic validation. Unknown fields are rejected, strings are trimmed,
  and tags are stored lowercase without duplicates.
* **passengers**: search the cleaned Titanic dataset, or export it as a feature matrix
  ready for model training.
* **stats**: five SQL analytics queries (GROUP BY, CTE with CASE, window functions, HAVING,
  and a JOIN).

Invalid input returns **422** with the field-level errors, and an unknown id returns **404**.
Set `APP_STORAGE=memory` to run the items API without a database; passengers and stats
need SQL Server.
"""

TAGS = [
    {"name": "items", "description": "Create, read, update and delete items"},
    {"name": "passengers", "description": "Titanic passengers stored in SQL Server"},
    {"name": "stats", "description": "Analytics queries over the passengers table"},
    {"name": "system", "description": "Operational endpoints"},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Week 1 AI Backend", version="0.2.0", description=DESCRIPTION, openapi_tags=TAGS
    )

    if settings.storage == "sql":
        db = Database(settings.db.conn_str())
        app.state.db = db
        app.state.item_store = SqlItemRepository(db)
        app.state.passenger_repo = PassengerRepository(db)
        app.include_router(passengers.router)
        app.include_router(stats.router)
    else:
        app.state.db = None
        app.state.item_store = InMemoryItemRepository()

    app.include_router(items.router)
    app.include_router(system.router)

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    return app


app = create_app()
