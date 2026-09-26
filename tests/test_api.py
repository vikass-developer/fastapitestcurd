import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_crud_flow(client: TestClient) -> None:
    created = client.post("/items", json={"name": "Pen", "price": 1.5, "tags": ["office"]})
    assert created.status_code == 201
    item_id = created.json()["id"]

    assert client.get(f"/items/{item_id}").json()["name"] == "Pen"
    assert len(client.get("/items", params={"tag": "office"}).json()) == 1

    updated = client.patch(f"/items/{item_id}", json={"price": 2.0})
    assert updated.status_code == 200
    assert updated.json()["price"] == 2.0
    assert updated.json()["updated_at"] is not None

    assert client.delete(f"/items/{item_id}").status_code == 204
    assert client.get(f"/items/{item_id}").status_code == 404


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/items", json={"name": "", "price": -1}).status_code == 422
    assert client.patch("/items/1", json={}).status_code == 400
    assert client.delete("/items/999").status_code == 404
