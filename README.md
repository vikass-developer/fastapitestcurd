# Week 1 — Python for AI

Two deliverables in one clean, installable Python repo:

| Topic | Deliverable | Where |
|---|---|---|
| Python refresh: typing, functions, classes, async, virtualenv, packaging | Small FastAPI CRUD service | [`src/app`](src/app) |
| NumPy, pandas, data cleaning, JSON/CSV handling | Cleaned public dataset (Titanic) + 5 reusable data utilities | [`src/datautils`](src/datautils), [`data/`](data) |

## Project layout

```
.
├── pyproject.toml          # packaging, deps, console script, pytest + ruff config
├── src/
│   ├── app/                # FastAPI service
│   │   ├── main.py         # app factory + routes
│   │   ├── models.py       # Pydantic schemas
│   │   └── repository.py   # async in-memory repository (asyncio.Lock)
│   └── datautils/          # data-cleaning library
│       ├── cleaning.py     # the 5 utilities
│       ├── io.py           # CSV / JSON / JSONL read & write
│       └── cli.py          # `clean-dataset` pipeline for Titanic
├── data/
│   ├── raw/titanic.csv
│   └── processed/          # titanic_clean.csv, titanic_clean.json
└── tests/                  # pytest suite for API and utilities
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## 1. FastAPI service

```bash
uvicorn app.main:app --reload --app-dir src
# open http://127.0.0.1:8000/docs
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/items?skip=&limit=&tag=` | List items (paging + tag filter) |
| GET | `/items/{id}` | Get one item |
| POST | `/items` | Create item |
| PATCH | `/items/{id}` | Partial update |
| DELETE | `/items/{id}` | Delete item |

**OpenAPI docs:** Swagger UI at `/docs`, ReDoc at `/redoc`, and the raw schema at `/openapi.json`. Every endpoint has a summary, a tag, example request bodies, and documented 400/404/422 responses.

**Validation (Pydantic v2):**
- `name` is 1–100 characters, `description` up to 500, and `price` must be greater than 0 and at most 1,000,000. You can send up to 10 `tags`.
- Strings are trimmed, and tags are stored lowercase without duplicates. The `?tag=` filter ignores case.
- Unknown fields are rejected with 422.
- A PATCH may leave fields out, but sending `null` for `name`, `price` or `tags` returns 422. Only `description` can be cleared.
- An empty PATCH body returns 400, a missing id returns 404, and an id below 1 returns 422.

Python concepts covered: type hints and `Annotated`, Pydantic models, classes (`ItemRepository`, a custom exception), `async`/`await` with `asyncio.Lock`, dependency injection, an app factory, and `pyproject.toml` packaging.

## 2. Data cleaning utilities

The five utilities in `datautils.cleaning` are pure (they never mutate their input), so they chain with `DataFrame.pipe`:

| # | Function | What it does |
|---|---|---|
| 1 | `standardize_columns(df)` | Column names to `snake_case` |
| 2 | `clean_strings(df)` | Trims and collapses whitespace; turns `''`, `N/A`, `?`, `null` into NaN |
| 3 | `fill_missing(df, numeric=..., overrides=...)` | Numeric columns get the median/mean/zero, text columns the mode, plus per-column overrides |
| 4 | `clip_outliers(df, columns, k=1.5)` | Caps values at the IQR (Tukey) fences with NumPy |
| 5 | `missing_report(df)` | Per-column dtype, missing count and %, and unique values |

`datautils.io` adds `read_table` / `write_table`, which pick CSV, JSON or JSONL from the file extension. Nested JSON is flattened with `json_normalize`.

### Clean the Titanic dataset

```bash
clean-dataset            # or: python -m datautils.cli data/raw/titanic.csv data/processed
```

Source: [datasciencedojo/datasets — titanic.csv](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) (891 rows).

| | Before | After |
|---|---|---|
| Missing `Age` | 177 (19.9%) | 0 (median fill) |
| Missing `Cabin` | 687 (77.1%) | column dropped |
| Missing `Embarked` | 2 | 0 (mode fill) |
| `Fare` outliers | max 512.33 | capped at Q3 + 3×IQR |
| Column names | `PassengerId`, `SibSp`… | `passenger_id`, `sib_sp`… |

Output: `data/processed/titanic_clean.csv` and `data/processed/titanic_clean.json`.

## Tests and lint

```bash
pytest
ruff check src tests
```
