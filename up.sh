#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
INGEST_OUTPUT="data/raw/documents.jsonl"
BM25_INDEX_FILE="data/index/bm25/bm25_index.pkl"
VECTOR_INDEX_FILE="data/index/vector/faiss.index"

BACKEND_URL="http://localhost:8000"
BACKEND_DOCS_URL="http://localhost:8000/docs"
FRONTEND_URL="http://localhost:5173"

UVICORN_PID=""
FRONTEND_PID=""
CLEANED_UP=0

cleanup() {
  local status=${1:-$?}

  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  echo
  echo "Shutting down services..."

  if [[ -n "$UVICORN_PID" ]]; then
    kill "$UVICORN_PID" 2>/dev/null || true
  fi

  if [[ -n "$FRONTEND_PID" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ -n "$UVICORN_PID" ]]; then
    wait "$UVICORN_PID" 2>/dev/null || true
  fi

  if [[ -n "$FRONTEND_PID" ]]; then
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "Exited with status $status"
  fi
}

trap 'cleanup 130; exit 130' INT
trap 'cleanup 143; exit 143' TERM
trap 'cleanup $?' EXIT

create_venv_if_missing() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Git Bash on Windows
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
  elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # Linux / macOS / WSL
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  else
    echo "Could not find activation script in $VENV_DIR"
    exit 1
  fi
}

install_requirements() {
  echo "Installing Python dependencies"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
}

run_ingest_if_missing() {
  if [[ -f "$INGEST_OUTPUT" ]]; then
    return
  fi

  echo "Ingest artifact missing: $INGEST_OUTPUT"
  echo "Running ingestion from ./docs"

  mkdir -p "$(dirname "$INGEST_OUTPUT")"
  python app/ingest.py docs "$INGEST_OUTPUT" || {
    echo "WARNING: ingestion failed; continuing startup"
    return
  }
}

run_index_build_if_missing() {
  if [[ -f "$BM25_INDEX_FILE" && -f "$VECTOR_INDEX_FILE" ]]; then
    return
  fi

  echo "Index artifacts missing"
  echo "Building BM25 and vector indexes"

  python - <<'PY' || {
import json
from pathlib import Path

from backend.app.search.bm25 import BM25Index
from backend.app.search.vector import VectorIndex

jsonl_path = Path("data/raw/documents.jsonl")

if not jsonl_path.exists():
    print("WARNING: data/raw/documents.jsonl not found; skipping index build")
    raise SystemExit(0)

documents = []
with open(jsonl_path, "r", encoding="utf-8") as file:
    for line in file:
        row = line.strip()
        if not row:
            continue
        payload = json.loads(row)
        doc_id = payload.get("doc_id")
        if not doc_id:
            continue
        documents.append(
            {
                "doc_id": doc_id,
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
            }
        )

if not documents:
    print("WARNING: no documents found in data/raw/documents.jsonl; skipping index build")
    raise SystemExit(0)

bm25 = BM25Index(index_dir="data/index/bm25")
bm25.build(documents)
bm25.save()

vector = VectorIndex(index_dir="data/index/vector")
vector.build(documents)
vector.save()

print("Index build complete")
PY
    echo "WARNING: index build failed; backend may return 503 for search"
  }
}

start_backend() {
  echo "Starting backend on $BACKEND_URL"
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
  UVICORN_PID=$!
}

start_frontend() {
  if [[ -f "frontend/package.json" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "frontend/package.json found, but npm is not installed"
      exit 1
    fi

    echo "Starting frontend dev server from ./frontend"
    (
      cd frontend
      npm install
      npm run dev -- --host 0.0.0.0 --port 5173
    ) &
    FRONTEND_PID=$!
  else
    echo "No frontend/package.json found; serving ./frontend as static files"
    python -m http.server 5173 --directory frontend &
    FRONTEND_PID=$!
  fi
}

main() {
  create_venv_if_missing
  activate_venv
  install_requirements

  run_ingest_if_missing
  run_index_build_if_missing

  start_backend
  start_frontend

  echo
  echo "Backend:  $BACKEND_URL"
  echo "API docs: $BACKEND_DOCS_URL"
  echo "Frontend: $FRONTEND_URL"
  echo "Press Ctrl+C to stop both servers"

  wait "$UVICORN_PID" "$FRONTEND_PID"
}

main
