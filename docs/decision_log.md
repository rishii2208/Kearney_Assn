# Decision Log

Date: 2026-03-13

## Scope and Constraints

This project is designed for local/assignment-style usage with:

- CPU-only inference (no GPU dependency)
- lightweight deployment (single machine)
- moderate corpus size and request volume
- quick iteration over retrieval quality

The decisions below optimize for reliability, simplicity, and predictable performance under those constraints.

---

## 1) Default fusion: Min-max score normalization (weighted hybrid) over RRF

### Decision

Use min-max normalized weighted fusion as the default:

`hybrid_score = alpha * bm25_norm + (1 - alpha) * vector_norm`

Keep Reciprocal Rank Fusion (RRF) implemented as an alternative mode, but not the default.

### Why this was chosen

- **Direct control of lexical vs semantic balance**: `alpha` has clear meaning and can be tuned for product behavior.
- **Uses score magnitude, not only rank order**: if one retriever is strongly confident, that confidence can influence final ranking.
- **Better for diagnostics**: output includes `bm25_score`, `vector_score`, and `hybrid_score`, making ranking behavior explainable.
- **Works well with existing API/UI design**: the system already exposes per-source scores to users and logs.

### Why RRF is not default

- RRF is robust and simple, but it ignores score magnitude and keeps only rank position information.
- For this use case, we wanted a tunable and interpretable blend rather than a rank-only combiner.

### Trade-offs accepted

- Min-max can be sensitive to narrow or skewed score ranges.
- RRF can be more stable across heterogeneous scorers.

### Mitigation

- Keep RRF available as a runtime alternative (`method="rrf"`) for A/B comparisons.
- Monitor eval metrics (`ndcg@10`, `recall@10`, `mrr@10`) and switch default if RRF consistently wins.

### Revisit trigger

Change default to RRF if score-scale instability causes ranking regressions or if offline eval shows sustained lift from RRF across representative query sets.

---

## 2) Embedding model: `all-MiniLM-L6-v2` over larger models

### Decision

Use `all-MiniLM-L6-v2` as the default embedding model.

### Why this was chosen

- **CPU-first latency**: significantly faster encoding on CPU than larger transformer models.
- **Smaller memory/index footprint**: lower-dimensional vectors reduce RAM and FAISS index size.
- **Strong quality/latency ratio**: generally good semantic retrieval quality for broad, Wikipedia-style text while keeping response times practical.
- **Operational simplicity**: easier local startup and fewer resource-related failures.

### Why larger models were not default

- Better semantic quality is possible, but with higher CPU cost, slower indexing/querying, and larger memory pressure.
- For this use case, those costs were not justified as the baseline default.

### Trade-offs accepted

- Potentially lower ceiling on semantic nuance vs large embedding models.

### Mitigation

- Model is configurable in `VectorIndex`.
- Startup validation checks model name and embedding dimension against saved metadata to prevent silent mismatch bugs.

### Revisit trigger

Adopt a larger model if offline eval shows meaningful quality gains (for example, consistent lift in `ndcg@10`/`mrr@10`) that justify higher latency and infrastructure cost.

---

## 3) Persistence for logs/metrics: SQLite over Postgres

### Decision

Use SQLite (`data/metrics/search_logs.db`) for request logging and metrics reads.

### Why this was chosen

- **Zero infrastructure overhead**: no separate DB service required.
- **Fast local setup**: ideal for assignment/demo and single-node deployment.
- **Workload fit**: append-heavy logs and simple aggregations (counts, percentiles, grouped query stats).
- **Good enough concurrency for current scale**: WAL mode + thread-safe access pattern cover expected traffic.

### Why Postgres was not default

- Adds operational complexity (service provisioning, credentials, migrations, backups).
- Overkill for current traffic profile and feature scope.

### Trade-offs accepted

- SQLite is not ideal for multi-node writes, high write concurrency, or complex analytical workloads at scale.

### Mitigation

- Keep DB logic isolated in `app/db.py`, including migrations and query helpers, so migration to Postgres later is straightforward.

### Revisit trigger

Migrate to Postgres when write concurrency, multi-instance deployment, retention requirements, or analytics complexity exceed SQLite’s practical limits.

---

## Verification Strategy

These decisions are validated continuously through:

- endpoint tests (`/search`, `/metrics`, `/logs`, `/experiments`)
- hybrid search tests (weighting behavior + edge cases)
- migration tests for logging schema evolution
- offline eval pipeline (`app/eval.py`) tracking retrieval metrics over time
