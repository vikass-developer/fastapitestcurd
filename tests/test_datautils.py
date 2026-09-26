import numpy as np
import pandas as pd

from datautils import (
    clean_strings,
    clip_outliers,
    fill_missing,
    missing_report,
    read_table,
    standardize_columns,
    write_table,
)


def test_standardize_columns() -> None:
    df = pd.DataFrame(columns=["PassengerId", " Ticket No. ", "home-city"])
    assert list(standardize_columns(df).columns) == ["passenger_id", "ticket_no", "home_city"]


def test_clean_strings() -> None:
    df = pd.DataFrame({"name": ["  Alice  ", "N/A", "Bob   Smith", "?"]})
    out = clean_strings(df)
    assert out["name"].tolist()[0] == "Alice"
    assert out["name"].tolist()[2] == "Bob Smith"
    assert out["name"].isna().sum() == 2
    assert df["name"].iloc[0] == "  Alice  "  # input not mutated


def test_fill_missing() -> None:
    df = pd.DataFrame({"age": [10.0, np.nan, 30.0], "city": ["A", None, "A"], "x": [None, 1, 2]})
    out = fill_missing(df, overrides={"x": -1})
    assert out["age"].tolist() == [10.0, 20.0, 30.0]
    assert out["city"].tolist() == ["A", "A", "A"]
    assert out["x"].iloc[0] == -1


def test_clip_outliers() -> None:
    df = pd.DataFrame({"v": [1, 2, 3, 4, 1000]})
    assert clip_outliers(df)["v"].max() < 1000


def test_missing_report() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
    report = missing_report(df)
    assert report.index[0] == "a"
    assert report.loc["a", "missing_pct"] == 50.0


def test_csv_json_roundtrip(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for name in ("t.csv", "t.json", "t.jsonl"):
        path = write_table(df, tmp_path / name)
        pd.testing.assert_frame_equal(read_table(path), df)
