# Break/Fix Log

Date: 2026-03-13

## Scenario A — Embedding model changed without rebuilding vector index

### What changed

I changed the embedding model from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` but did not rebuild the FAISS index artifacts.

### What error showed up

At startup, vector index loading failed with a model mismatch error:

- `ValueError: Model mismatch: index was built with 'all-MiniLM-L6-v2' but current config expects 'all-mpnet-base-v2'. Please rebuild the index ...`

In mismatch cases where model name could pass but dimensions differed, the dimension check would also fail with:

- `ValueError: Dimension mismatch: index has dim=<saved_dim> but model 'all-mpnet-base-v2' produces dim=<expected_dim>. Please rebuild the index ...`

### How startup validation caught it

The startup path called `VectorIndex.load()` via FastAPI lifespan initialization. `load()` read `metadata.json` and validated:

1. saved `model_name` vs configured runtime model name
2. saved embedding `dim` vs runtime model embedding dimension

Because artifacts were stale, validation failed before serving traffic.

### Root cause

I changed runtime model configuration but left old vector artifacts in `data/index/vector` built by a different model.

### Fix applied

I rebuilt vector artifacts with the current model (`all-mpnet-base-v2`) using the normal build + save flow.

### How I verified the fix

I verified all of the following:

- `data/index/vector/metadata.json` contained the expected model name and dimension
- backend startup completed without vector validation exceptions
- `GET /health` returned `200`
- `POST /search` returned valid ranked results instead of failing during index load

---

## Scenario B — NOT NULL column added to `search_logs` without migration

### What changed

I introduced a new `NOT NULL` column in `search_logs` directly, without adding a migration step and without updating inserts atomically.

### What broke

Search logging writes started failing after restart. The API hit SQLite integrity errors on insert (for example `NOT NULL constraint failed`) because existing insert paths did not provide the new required value.

### Root cause

Schema and application insert behavior drifted out of sync due to unmanaged DDL change.

### Fix applied

I used the migration system already implemented in `app/db.py`:

- schema changes were moved into explicit versioned migrations
- startup (`init_db()`) applied pending migrations automatically
- table evolution stayed compatible with runtime insert behavior

### How I verified the fix

I verified that:

- startup applied migrations successfully (`schema_migrations` tracked the applied version)
- inserts through `log_request()` succeeded after restart
- existing rows remained readable
- migration tests passed for old-schema-to-new-schema upgrade behavior

---

## Scenario C — Min-max normalization bug when all scores were equal

### What changed

The hybrid normalization path used plain min-max scaling without guarding the `max == min` case.

### What broke

When all scores in a source list were equal, normalization produced invalid values (division by zero behavior / NaN propagation in downstream scoring), ranking became unstable, and eval metrics dropped.

### Root cause

The normalization formula assumed non-zero score range and did not handle degenerate score distributions.

### Fix applied

I added the guard in `app/search/hybrid.py`:

- if `max == min`, normalization returned stable fallback values (`1.0` for positive equal scores, otherwise `0.0`)
- this prevented divide-by-zero/NaN propagation and preserved deterministic fusion behavior

### How I verified the fix

I verified that:

- hybrid tests covering equal-score input passed
- no divide-by-zero or NaN values appeared in fused output
- ranking output remained deterministic
- eval metrics recovered from the regression pattern seen during the break
