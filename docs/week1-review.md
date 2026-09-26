# Week 1 Review — Python for AI

## Weekly summary

| Day | Topic | What was built | Key takeaway |
|---|---|---|---|
| 1 | Python refresh | FastAPI CRUD service: typed Pydantic models, an async repository with `asyncio.Lock`, dependency injection, an app factory, and `pyproject.toml` packaging with a src layout | Type hints are how FastAPI does its work: they drive validation, serialization and the OpenAPI docs |
| 2 | NumPy + pandas | 5 pure cleaning functions chained with `DataFrame.pipe`, CSV/JSON/JSONL I/O, and the Titanic dataset cleaned from 866 missing values to 0 | Keep cleaning functions pure (return a new frame and never mutate the input), so they're easy to test and reuse |
| 3 | SQL + database from Python | SQL Server schema with constraints and indexes, 5 analytics queries, and a repository layer over `pyodbc` | Parameterize every value. Keep SQL in one place, and let routes depend on a repository interface rather than on SQL |
| 4 | Mini-project | One API over items (CRUD in SQL Server), passenger search, 5 `/stats` endpoints, and a `/passengers/features` export ready for model training | The same data serves an operational API, analytics, and a feature matrix |
| 5 | Review + refactor | The refactor below, 35 tests across memory and SQL storage, and these notes | Refactor behind tests: the same tests ran against both item stores |

### Final numbers

- **35 tests** pass, and `ruff` is clean. The items tests run against both storage backends through a parametrized fixture.
- The survival rates computed in SQL match the published Titanic figures, for example 96.8% for 1st-class women and 13.5% for 3rd-class men.

## Refactor log

| Before | After | Why |
|---|---|---|
| All routes in `main.py` | `routers/` (items, passengers, stats, system) | Each file has one job; `main.py` only wires the app together |
| `ItemRepository` concrete class used directly | `ItemStore` **Protocol** with `InMemoryItemRepository` and `SqlItemRepository` | Swap storage without touching routes; tests exercise both |
| `ItemNotFoundError(item_id)` | `NotFoundError(resource, key)` | One 404 handler for items and passengers |
| Repository created inside `create_app` | `Settings.from_env()` + `create_app(settings)` | Configured from the environment; tests inject settings instead of patching |
| `Repo` type alias defined inside `create_app` | Dependencies at module level in `dependencies.py` | Pylance flagged it, and module-level aliases are reusable |
| Prices stored as given | Rounded to 2 decimals in the model | The memory store and SQL `DECIMAL(12,2)` now return the same value |

## Bugs found during the week (and fixes)

1. **PATCH `{"name": null}` saved an item with no name.** `model_copy(update=...)` does not re-validate. The fix is a validator that allows a field to be omitted but rejects an explicit `null`, plus a regression test.
2. **Every request returned 422 at first.** `from __future__ import annotations` turned a type alias defined inside a function into a string that FastAPI couldn't resolve, so it treated the dependency as a request field. The fix was to remove the import from that module; the aliases later moved to module level.
3. **`--reload` ignored file edits.** File-change events don't arrive reliably in a OneDrive-synced folder. The fix is `WATCHFILES_FORCE_POLLING=true`.
4. **Route order matters.** `/passengers/features` has to be declared before `/passengers/{passenger_id}`, or `"features"` is parsed as an id and returns 422.
5. **The LIKE search trusted user wildcards.** `name=%%` matched every row. The fix escapes `%`, `_` and `[` and uses `ESCAPE '\'`.

## PostgreSQL vs SQL Server

The tracker names PostgreSQL. This project uses SQL Server because it's installed locally. What changes when porting:

| Concept | SQL Server (used here) | PostgreSQL |
|---|---|---|
| Driver | `pyodbc` with `?` placeholders | `psycopg` with `%s` placeholders |
| Auto-increment key | `INT IDENTITY(1,1)` | `INT GENERATED ALWAYS AS IDENTITY` |
| Paging | `OFFSET n ROWS FETCH NEXT m ROWS ONLY` | `LIMIT m OFFSET n` |
| Return the new id | `OUTPUT INSERTED.id` | `RETURNING id` |
| Boolean | `BIT` (needs `CAST(... AS INT)` to `SUM`) | `BOOLEAN` (`COUNT(*) FILTER (WHERE survived)`) |
| Rows to JSON | `FOR JSON PATH` | `json_agg(...)` |
| Create-if-missing | `IF OBJECT_ID(...) IS NULL CREATE TABLE` | `CREATE TABLE IF NOT EXISTS` |
| UTC timestamp | `SYSUTCDATETIME()` into `DATETIME2` | `now()` into `TIMESTAMPTZ` |

The window functions, CTEs, `CASE`, `HAVING` and joins in the 5 queries are standard SQL and port unchanged.

## 10 interview questions

**1. What happens when a request hits a FastAPI endpoint with a Pydantic body?**
FastAPI reads the type hints on the function. It parses the JSON body into the Pydantic model, which validates types, lengths and ranges, and runs any custom validators. If anything fails, it returns 422 with the path and message for each field. The return value is then filtered through `response_model`. The same hints produce the OpenAPI schema at `/openapi.json`.

**2. `async def` or `def` for an endpoint that calls a blocking database driver?**
Never call blocking I/O directly inside `async def`, because that stalls the event loop for every request. There are two options. Declare the endpoint as a plain `def`, and FastAPI runs it in a threadpool. Or keep it `async` and offload the blocking call with `await asyncio.to_thread(...)`, which is what this repo does around `pyodbc`. The fully async option is an async driver such as `asyncpg`, `psycopg` async, or `aioodbc`.

**3. Why use a repository layer, and why a `Protocol` instead of a base class?**
A repository hides storage details, so routes stay the same when the storage changes; here, memory and SQL Server are swapped with an env var. A `Protocol` gives structural typing: a class qualifies by having the right methods, with no inheritance needed. That keeps implementations decoupled and makes test fakes trivial.

**4. How do you prevent SQL injection when the WHERE clause is built dynamically?**
Values always go in as bound parameters (`?`), never formatted into the SQL string. When identifiers such as column names have to be dynamic, take them from a fixed whitelist that the code controls. In `SqlItemRepository._update`, the columns come from the Pydantic model's fields and are checked against `_UPDATABLE_COLUMNS`. `LIKE` patterns also need their wildcards escaped.

**5. `WHERE` vs `HAVING`?**
`WHERE` filters rows before grouping. `HAVING` filters groups after aggregation, so it can use `COUNT(*)` and similar. Query 4 uses `HAVING COUNT(*) >= ?` to drop family sizes with too few passengers to be meaningful.

**6. How do you get the top N rows per group in SQL?**
Use a window function: `ROW_NUMBER() OVER (PARTITION BY group ORDER BY metric DESC)` in a CTE, then filter `WHERE rn <= N` (Query 3). Pick `RANK()` or `DENSE_RANK()` instead if ties should share a position. `GROUP BY` can't do this, because it collapses the rows.

**7. How did you handle missing values, and why the median for `Age`?**
First measure the gaps with `missing_report`. `Cabin` was 77% missing, so it was dropped, because imputing it would invent data. `Age` was 20% missing and filled with the **median**, which is robust to skew and outliers, unlike the mean. `Embarked` had 2 missing values, filled with the **mode**. In a real ML pipeline, you compute the fill values on the training split only, to avoid leaking information from the test set.

**8. What does `build_features` do, and why one-hot encode `pclass` when it's already a number?**
It turns cleaned rows into a purely numeric matrix: one-hot class and port, binary sex, family size, is-alone, and the target in the last column. `pclass` is a category, not a quantity. Leaving it as 1/2/3 would tell a linear model that class 3 is "three times" class 1. One-hot encoding removes that false ordering.

**9. How do the tests hit a real database without breaking the dev data?**
A session-scoped fixture creates a separate `week1_ai_test` database, applies the same `schema.sql`, loads the CSV, and drops the database at the end. Each test clears the `items` table first. The `client` fixture is parametrized as `["memory", "sql"]`, so every items test runs against both implementations, which proves they behave the same. If SQL Server is unreachable, those tests are skipped instead of failing.

**10. What changes to take this service to production?**
- Run schema changes as versioned migrations (Alembic) instead of an idempotent script.
- Use a proper connection pool or an async driver, with timeouts.
- Handle secrets with a secret manager instead of env defaults.
- Add authentication and rate limiting.
- Log in a structured format with request ids.
- Split `/health` into liveness and readiness checks.
- Run CI with the test suite against a containerized database.
- Paginate with total counts or a cursor.
- Add a `Dockerfile` so the database and the app run together.
