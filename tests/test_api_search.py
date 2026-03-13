import json
from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient

# Allow importing backend.app.main even when optional vector deps are not installed.
if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.ModuleType("faiss")

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class _SentenceTransformerStub:
        def __init__(self, *args, **kwargs):
            pass

    sentence_transformers_stub.SentenceTransformer = _SentenceTransformerStub
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from backend.app.main import app
from backend.app.routes import get_bm25_index, get_vector_index


class MockIndex:
    def __init__(self, index_dir: Path, results):
        self.index_dir = index_dir
        self._results = list(results)

    def query(self, query: str, k: int = 10):
        return self._results[:k]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("backend.app.routes.log_request", lambda **kwargs: "test-request-id")

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _write_documents_jsonl(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    docs = [
        {
            "doc_id": "doc-python",
            "title": "Python",
            "text": "Python is a high-level programming language used for data and web APIs.",
        },
        {
            "doc_id": "doc-ml",
            "title": "Machine Learning",
            "text": "Machine learning models are trained on data to make predictions.",
        },
    ]

    with open(index_dir / "documents.jsonl", "w", encoding="utf-8") as file:
        for doc in docs:
            file.write(json.dumps(doc) + "\n")


def test_search_valid_request_returns_200_with_scores(client, tmp_path):
    bm25_dir = tmp_path / "bm25"
    _write_documents_jsonl(bm25_dir)

    bm25 = MockIndex(
        index_dir=bm25_dir,
        results=[
            {"doc_id": "doc-python", "score": 5.0},
            {"doc_id": "doc-ml", "score": 1.0},
        ],
    )
    vector = MockIndex(
        index_dir=tmp_path / "vector",
        results=[
            {"doc_id": "doc-python", "score": 0.9},
            {"doc_id": "doc-ml", "score": 0.2},
        ],
    )

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "python", "top_k": 2, "alpha": 0.5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload
    assert len(payload["results"]) > 0

    first = payload["results"][0]
    assert first["doc_id"] == "doc-python"
    assert "bm25_score" in first
    assert "vector_score" in first
    assert "hybrid_score" in first
    assert "snippet" in first


def test_search_empty_query_returns_422(client, tmp_path):
    bm25 = MockIndex(index_dir=tmp_path / "bm25", results=[])
    vector = MockIndex(index_dir=tmp_path / "vector", results=[])

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "", "top_k": 10, "alpha": 0.5},
    )

    assert response.status_code == 422


def test_search_alpha_out_of_range_returns_422(client, tmp_path):
    bm25 = MockIndex(index_dir=tmp_path / "bm25", results=[])
    vector = MockIndex(index_dir=tmp_path / "vector", results=[])

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "python", "top_k": 10, "alpha": 1.5},
    )

    assert response.status_code == 422


def test_search_missing_indexes_returns_503(client):
    app.dependency_overrides[get_bm25_index] = lambda: None
    app.dependency_overrides[get_vector_index] = lambda: None

    response = client.post(
        "/search",
        json={"query": "python", "top_k": 10, "alpha": 0.5},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Search indexes are not loaded"


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "commit" in payload
