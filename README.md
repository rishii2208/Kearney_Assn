# Kearney Assignment

Welcome to My Project Mr. Evaluator !!!


## 1) Run the app

### Fast path (recommended)

Use the startup script:

```bash
bash up.sh
```

What it does:

- creates `.venv` if missing
- installs Python + frontend deps
- runs ingest/index build if artifacts are missing
- starts backend + frontend

URLs:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Manual run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

If your vector index artifacts are missing, search endpoint will return `503` until indexes are built.

## 2) Run tests

### All tests

```bash
python -m pytest -q
```

### Verbose

```bash
python -m pytest -v
```

## 3) Run eval

Eval CLI:

```bash
python app/eval.py <queries.jsonl> <qrels.json>
```

Example:

```bash
python app/eval.py data/eval/queries.jsonl data/eval/qrels.json
```

Optional flags:

```bash
python app/eval.py data/eval/queries.jsonl data/eval/qrels.json \
  --bm25-index-dir data/index/bm25 \
  --vector-index-dir data/index/vector \
  --output-csv data/metrics/experiments.csv
```

What eval does:

- runs hybrid search at `alpha=0.5`
- computes `ndcg@10`, `recall@10`, `mrr@10`
- appends one row to `data/metrics/experiments.csv` with timestamp + git commit

## 4) Minimal input formats

`queries.jsonl` (one JSON object per line):

```json
{"query_id": "q001", "query_text": "python web framework"}
{"query_id": "q002", "query_text": "roman empire fall"}
```

`qrels.json` (map query_id -> {doc_id: relevance}):

```json
{
  "q001": { "doc_1": 2, "doc_9": 1 },
  "q002": { "doc_4": 1 }
}
```
