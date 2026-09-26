# Week 1 — Python for AI

A small backend with data ready for AI work: **FastAPI + SQL Server + pandas**, in one installable Python repo.

| Day | Topic | Deliverable | Where |
|---|---|---|---|
| 1 | Python refresh: typing, functions, classes, async, virtualenv, packaging | Small FastAPI service + clean Python repo | [`src/app`](src/app) |
| 2 | NumPy, pandas, data cleaning, JSON/CSV handling | Cleaned public dataset + 5 reusable data utilities | [`src/datautils`](src/datautils), [`data/`](data) |
| 3 | SQL basics + a relational database from Python | 5 queries + repository layer | [`src/db`](src/db), [`src/app/repositories`](src/app/repositories) |
| 4 | Mini-project: Python + API + DB | Small data/AI-ready backend project | the whole app ([endpoints](#api-endpoints)) |
| 5 | Review + refactor + notes | Weekly summary + 10 interview questions | [`docs/week1-review.md`](docs/week1-review.md) |

> **Database:** the tracker says PostgreSQL, but this project uses **Microsoft SQL Server 2022** through `pyodbc`, because that is what is installed locally. The SQL concepts are the same; the T-SQL differences are covered in the [review notes](docs/week1-review.md#postgresql-vs-sql-server).

## Project layout

```
.
├── pyproject.toml              # packaging, deps, console scripts, pytest + ruff config
├── src/
│   ├── app/                    # FastAPI application
│   │   ├── main.py             # app factory: picks SQL or in-memory storage
│   │   ├── config.py           # Settings from env (APP_STORAGE)
│   │   ├── dependencies.py     # Depends() providers for the repositories
│   │   ├── models.py           # Pydantic request/response schemas
│   │   ├── repositories/       # repository layer
│   │   │   ├── base.py         # ItemStore Protocol + NotFoundError
│   │   │   ├── memory.py       # in-memory item store (asyncio.Lock)
│   │   │   ├── sql_items.py    # SQL Server item store (items + item_tags)
│   │   │   └── passengers.py   # passenger search + the 5 analytics queries
│   │   └── routers/            # items, passengers, stats, system
│   ├── db/                     # database layer, no FastAPI imports
│   │   ├── config.py           # DbSettings from env (MSSQL_*)
│   │   ├── connection.py       # pyodbc wrapper: transactions, rows as dicts
│   │   ├── schema.sql          # tables, constraints, indexes (idempotent)
│   │   ├── queries.py          # the 5 analytics queries
│   │   └── init_db.py          # `init-db`: create DB, apply schema, load CSV
│   └── datautils/              # pandas utilities
│       ├── cleaning.py         # the 5 cleaning utilities
│       ├── features.py         # feature matrix ready for model training
│       ├── io.py               # CSV / JSON / JSONL read & write
│       └── cli.py              # `clean-dataset` pipeline
├── data/raw/ + data/processed/ # Titanic: raw and cleaned
├── docs/week1-review.md        # weekly summary + interview questions
└── tests/                      # 35 tests; the SQL ones use a throwaway database
```

## Setup

Requirements: Python 3.11+, SQL Server (any edition), and **ODBC Driver 17 or 18 for SQL Server**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

clean-dataset     # data/raw/titanic.csv -> data/processed/titanic_clean.{csv,json}
init-db           # creates database week1_ai, applies schema.sql, loads 891 passengers
uvicorn app.main:app --reload --app-dir src
# open http://127.0.0.1:8000/docs
```

Configuration comes from environment variables:

| Variable | Default | Notes |
|---|---|---|
| `APP_STORAGE` | `sql` | `memory` runs the items API with no database; passengers and stats are then disabled |
| `MSSQL_SERVER` | `localhost` | e.g. `localhost\SQLEXPRESS` |
| `MSSQL_DATABASE` | `week1_ai` | |
| `MSSQL_DRIVER` | `ODBC Driver 17 for SQL Server` | |
| `MSSQL_AUTH` | `Trusted_Connection=yes` | Windows login; use `UID=sa;PWD=...` for a SQL login |

## API endpoints

Interactive docs: `/docs` (Swagger UI) and `/redoc`. The raw schema is at `/openapi.json`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check and database ping |
| GET | `/items?skip=&limit=&tag=` | List items (paging + tag filter) |
| GET | `/items/{id}` | Get one item |
| POST | `/items` | Create item |
| PATCH | `/items/{id}` | Partial update |
| DELETE | `/items/{id}` | Delete item |
| GET | `/passengers?pclass=&sex=&survived=&min_age=&max_age=&name=&skip=&limit=` | Search passengers |
| GET | `/passengers/{id}` | Get one passenger |
| GET | `/passengers/features?format=json\|csv` | Feature matrix ready for model training |
| GET | `/stats/survival-by-class-sex` | Query 1 |
| GET | `/stats/survival-by-age-group` | Query 2 |
| GET | `/stats/oldest-per-class?top=3` | Query 3 |
| GET | `/stats/survival-by-family-size?min_passengers=5` | Query 4 |
| GET | `/stats/embarked` | Query 5 |

**Validation (Pydantic v2):**
- `name` is 1–100 characters and `description` up to 500. `price` must be greater than 0 and at most 1,000,000, and is rounded to 2 decimals. You can send up to 10 `tags` of 1–50 characters each.
- Strings are trimmed, and tags are stored lowercase without duplicates. The `?tag=` filter ignores case.
- Unknown fields are rejected with 422.
- A PATCH may leave fields out, but sending `null` for `name`, `price` or `tags` returns 422. Only `description` can be cleared.
- An empty PATCH body returns 400, a missing id returns 404, and an id below 1 returns 422.
- Passenger search checks that `pclass` is 1–3, `sex` is `male` or `female`, and `min_age` is not greater than `max_age`. `%` and `_` in `name` are matched literally, not as wildcards.

## SQL: schema, 5 queries, repository layer

[`schema.sql`](src/db/schema.sql) creates `items`, `item_tags` (1-to-many, `ON DELETE CASCADE`) and `passengers`. It uses CHECK constraints and a covering index, and is safe to re-run.

The five queries in [`db/queries.py`](src/db/queries.py) each use a different SQL building block:

| # | Query | SQL concepts | Result on the cleaned data |
|---|---|---|---|
| 1 | Survival by class and sex | `GROUP BY` on 2 columns, `COUNT`/`SUM`/`AVG` | 1st-class women 96.8%, 3rd-class men 13.5% |
| 2 | Survival by age group | CTE, `CASE` bucketing, ordered groups | children 58.0%, seniors 26.9% |
| 3 | Oldest N per class | `ROW_NUMBER() OVER (PARTITION BY ...)` | oldest was 80, in 1st class, and survived |
| 4 | Survival by family size | `GROUP BY` an expression, `HAVING` | size 4 is the best at 72.4%; alone is 30.4% |
| 5 | Summary by port | `JOIN` to a `VALUES` lookup, `SUM() OVER ()` share | Southampton 72.5% of passengers |

**Repository layer:** routes depend on an `ItemStore` **Protocol**, not a concrete class. `InMemoryItemRepository` and `SqlItemRepository` both implement it, and `APP_STORAGE` picks one at startup. All SQL is parameterized (`?`). The only dynamic parts are column names, and those come from a whitelist. Blocking `pyodbc` calls run in `asyncio.to_thread` so they don't block the event loop, and every write runs in a transaction that rolls back on error.

## Data cleaning utilities

The five utilities in `datautils.cleaning` are pure (they never mutate their input), so they chain with `DataFrame.pipe`:

| # | Function | What it does |
|---|---|---|
| 1 | `standardize_columns(df)` | Column names to `snake_case` |
| 2 | `clean_strings(df)` | Trims and collapses whitespace; turns `''`, `N/A`, `?`, `null` into NaN |
| 3 | `fill_missing(df, numeric=..., overrides=...)` | Numeric columns get the median/mean/zero, text columns the mode, plus per-column overrides |
| 4 | `clip_outliers(df, columns, k=1.5)` | Caps values at the IQR (Tukey) fences with NumPy |
| 5 | `missing_report(df)` | Per-column dtype, missing count and %, and unique values |

`datautils.io` adds `read_table` / `write_table`, which pick CSV, JSON or JSONL from the file extension. `datautils.features.build_features` turns the cleaned rows into a numeric matrix ready for model training: one-hot class and port, `is_female`, `family_size`, `is_alone`, and the `survived` target.

Titanic source: [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) (891 rows).

| | Before | After |
|---|---|---|
| Missing `Age` | 177 (19.9%) | 0 (median fill) |
| Missing `Cabin` | 687 (77.1%) | column dropped |
| Missing `Embarked` | 2 | 0 (mode fill) |
| `Fare` outliers | max 512.33 | capped at Q3 + 3×IQR |
| Column names | `PassengerId`, `SibSp`… | `passenger_id`, `sib_sp`… |

## Tests and lint

```bash
pytest               # 35 tests; items tests run against both memory and SQL storage
ruff check src tests
```

The SQL tests create a throwaway `week1_ai_test` database, load it, and drop it at the end. If SQL Server can't be reached, they are skipped.

Week 1 days 3–5: SQL Server data layer, mini-project API, and weekly review.

- db/: schema, 5 analytics queries, init-db loader
- Repository layer: in-memory or SQL Server item store, chosen with APP_STORAGE
- New endpoints: /passengers search, /stats (5 queries), /passengers/features (JSON or CSV)
- Refactor: separate route files, settings from environment variables
- docs/week1-review.md: weekly summary + 10 interview questions
- 35 tests pass; the item tests run against both memory and SQL storage

