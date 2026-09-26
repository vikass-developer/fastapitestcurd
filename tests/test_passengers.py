"""Passenger search, feature export and the five stats queries (SQL Server only)."""

import csv
import io

from fastapi.testclient import TestClient


def test_search_filters(sql_client: TestClient) -> None:
    rows = sql_client.get(
        "/passengers", params={"pclass": 1, "sex": "female", "survived": False, "limit": 100}
    ).json()
    assert len(rows) == 3  # 94 first-class women, 91 survived
    assert all(r["pclass"] == 1 and r["sex"] == "female" and not r["survived"] for r in rows)


def test_search_age_range_and_name(sql_client: TestClient) -> None:
    rows = sql_client.get("/passengers", params={"min_age": 70, "max_age": 80}).json()
    assert rows and all(70 <= r["age"] <= 80 for r in rows)
    named = sql_client.get("/passengers", params={"name": "BARKWORTH"}).json()
    assert [r["passenger_id"] for r in named] == [631]


def test_search_treats_like_wildcards_literally(sql_client: TestClient) -> None:
    assert sql_client.get("/passengers", params={"name": "%%"}).json() == []


def test_search_validation(sql_client: TestClient) -> None:
    assert sql_client.get("/passengers", params={"pclass": 4}).status_code == 422
    assert sql_client.get("/passengers", params={"sex": "x"}).status_code == 422
    assert sql_client.get("/passengers", params={"min_age": 50, "max_age": 10}).status_code == 422


def test_get_passenger(sql_client: TestClient) -> None:
    assert sql_client.get("/passengers/1").json()["name"] == "Braund, Mr. Owen Harris"
    missing = sql_client.get("/passengers/99999")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Passenger 99999 not found"}


def test_features_json_and_csv(sql_client: TestClient) -> None:
    rows = sql_client.get("/passengers/features").json()
    assert len(rows) == 891
    assert list(rows[0])[-1] == "survived"
    assert {"pclass_1", "embarked_S", "is_female", "family_size"} <= rows[0].keys()

    resp = sql_client.get("/passengers/features", params={"format": "csv"})
    assert resp.headers["content-type"].startswith("text/csv")
    assert len(list(csv.DictReader(io.StringIO(resp.text)))) == 891


def test_stats_survival_by_class_sex(sql_client: TestClient) -> None:
    rows = sql_client.get("/stats/survival-by-class-sex").json()
    assert len(rows) == 6
    assert sum(r["passengers"] for r in rows) == 891
    first_female = next(r for r in rows if r["pclass"] == 1 and r["sex"] == "female")
    assert first_female["survival_rate_pct"] == 96.8


def test_stats_age_groups(sql_client: TestClient) -> None:
    rows = sql_client.get("/stats/survival-by-age-group").json()
    assert rows[0]["age_group"].startswith("child")
    assert sum(r["passengers"] for r in rows) == 891


def test_stats_oldest_per_class(sql_client: TestClient) -> None:
    rows = sql_client.get("/stats/oldest-per-class", params={"top": 2}).json()
    assert len(rows) == 6
    assert rows[0] == {
        "pclass": 1, "age_rank": 1, "passenger_id": 631,
        "name": "Barkworth, Mr. Algernon Henry Wilson", "age": 80.0, "survived": True,
    }


def test_stats_family_size_having(sql_client: TestClient) -> None:
    rows = sql_client.get("/stats/survival-by-family-size", params={"min_passengers": 20}).json()
    assert all(r["passengers"] >= 20 for r in rows)
    assert [r["family_size"] for r in rows] == [1, 2, 3, 4, 6]


def test_stats_embarked(sql_client: TestClient) -> None:
    rows = sql_client.get("/stats/embarked").json()
    assert [r["port_name"] for r in rows] == ["Southampton", "Cherbourg", "Queenstown"]
    assert abs(sum(r["share_pct"] for r in rows) - 100) < 0.2


def test_passenger_routes_absent_in_memory_mode() -> None:
    from app.config import Settings
    from app.main import create_app

    client = TestClient(create_app(Settings(storage="memory")))
    assert client.get("/passengers").status_code == 404
    assert "/stats/embarked" not in client.get("/openapi.json").json()["paths"]
