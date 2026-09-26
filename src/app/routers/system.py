import asyncio

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness and database check")
async def health(request: Request) -> dict[str, str]:
    db = request.app.state.db
    result = {"status": "ok", "storage": "memory" if db is None else "sql"}
    if db is not None:
        result["database"] = "ok" if await asyncio.to_thread(db.ping) else "unreachable"
    return result
