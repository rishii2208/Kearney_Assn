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
import backend.app.routes as routes
from backend.app.routes import get_bm25_index, get_vector_index


class MockIndex:
    def __init__(self, index_dir: Path, results):
        self.index_dir = index_dir
        self._results = list(results)

    def query(self, query: str, k: int = 10):
        return self._results[:k]


class RecordingIndex(MockIndex):
    def __init__(self, index_dir: Path, results):
        super().__init__(index_dir=index_dir, results=results)
        self.seen_queries = []

    def query(self, query: str, k: int = 10):
        self.seen_queries.append(query)
        return super().query(query, k)


@pytest.fixture
def client(monkeypatch):
    routes._SEARCH_RATE_LIMIT.clear()
    monkeypatch.setattr("backend.app.routes.log_request", lambda **kwargs: "test-request-id")

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    routes._SEARCH_RATE_LIMIT.clear()


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


def test_search_strips_leading_and_trailing_whitespace(client, tmp_path):
    bm25_dir = tmp_path / "bm25"
    _write_documents_jsonl(bm25_dir)

    bm25 = RecordingIndex(
        index_dir=bm25_dir,
        results=[{"doc_id": "doc-python", "score": 5.0}],
    )
    vector = RecordingIndex(
        index_dir=tmp_path / "vector",
        results=[{"doc_id": "doc-python", "score": 0.9}],
    )

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "   python   ", "top_k": 1, "alpha": 0.5},
    )

    assert response.status_code == 200
    assert bm25.seen_queries[-1] == "python"
    assert vector.seen_queries[-1] == "python"


def test_search_collapses_multiple_spaces(client, tmp_path):
    bm25_dir = tmp_path / "bm25"
    _write_documents_jsonl(bm25_dir)

    bm25 = RecordingIndex(
        index_dir=bm25_dir,
        results=[{"doc_id": "doc-python", "score": 5.0}],
    )
    vector = RecordingIndex(
        index_dir=tmp_path / "vector",
        results=[{"doc_id": "doc-python", "score": 0.9}],
    )

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "python    web   framework", "top_k": 1, "alpha": 0.5},
    )

    assert response.status_code == 200
    assert bm25.seen_queries[-1] == "python web framework"
    assert vector.seen_queries[-1] == "python web framework"


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


def test_search_whitespace_only_query_returns_422(client, tmp_path):
    bm25 = MockIndex(index_dir=tmp_path / "bm25", results=[])
    vector = MockIndex(index_dir=tmp_path / "vector", results=[])

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "     ", "top_k": 10, "alpha": 0.5},
    )

    assert response.status_code == 422
    assert "whitespace" in response.json()["detail"]


def test_search_special_characters_only_query_returns_422(client, tmp_path):
    bm25 = MockIndex(index_dir=tmp_path / "bm25", results=[])
    vector = MockIndex(index_dir=tmp_path / "vector", results=[])

    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_vector_index] = lambda: vector

    response = client.post(
        "/search",
        json={"query": "!!! ??? ###", "top_k": 10, "alpha": 0.5},
    )

    assert response.status_code == 422
    assert "letter or number" in response.json()["detail"]


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


def test_search_rate_limit_returns_429(client, tmp_path, monkeypatch):
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

    routes._SEARCH_RATE_LIMIT.clear()
    monkeypatch.setattr("backend.app.routes.RATE_LIMIT_MAX_REQUESTS", 2)

    body = {"query": "python", "top_k": 2, "alpha": 0.5}
    response_1 = client.post("/search", json=body)
    response_2 = client.post("/search", json=body)
    response_3 = client.post("/search", json=body)

    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_3.status_code == 429
    assert "Rate limit exceeded" in response_3.json()["detail"]


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "commit" in payload


def test_top_queries_endpoint_returns_grouped_counts(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.routes.get_top_queries",
        lambda limit=10: [
            {"query": "python", "count": 4, "last_seen": "2026-03-13T10:00:00+00:00"},
            {"query": "fastapi", "count": 2, "last_seen": "2026-03-13T09:30:00+00:00"},
        ][:limit],
    )

    response = client.get("/top-queries?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert "top_queries" in payload
    assert payload["top_queries"][0]["query"] == "python"
    assert payload["top_queries"][0]["count"] == 4


def test_zero_result_queries_endpoint_returns_grouped_counts(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.routes.get_zero_result_queries",
        lambda limit=10: [
            {"query": "obscure term", "count": 3, "last_seen": "2026-03-13T10:05:00+00:00"},
            {"query": "rare topic", "count": 1, "last_seen": "2026-03-13T09:00:00+00:00"},
        ][:limit],
    )

    response = client.get("/zero-result-queries?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert "zero_result_queries" in payload
    assert payload["zero_result_queries"][0]["query"] == "obscure term"
    assert payload["zero_result_queries"][0]["count"] == 3


def test_experiments_endpoint_returns_rows(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.routes._read_experiments_csv",
        lambda: [
            {
                "timestamp": "2026-03-13T11:00:00+00:00",
                "git_commit": "abc123",
                "ndcg_at_10": 0.4567,
                "recall_at_10": 0.789,
                "mrr_at_10": 0.6123,
            }
        ],
    )

    response = client.get("/experiments")

    assert response.status_code == 200
    payload = response.json()
    assert "experiments" in payload
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["git_commit"] == "abc123"


def test_logs_endpoint_returns_rows_with_filters(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.routes.get_logs_filtered",
        lambda limit, start_created_at, end_created_at, severity: [
            {
                "request_id": "req-1",
                "query": "python",
                "latency_ms": 12.3,
                "top_k": 10,
                "alpha": 0.5,
                "result_count": 2,
                "error": "timeout",
                "created_at": "2026-03-13T12:00:00+00:00",
                "user_agent": None,
            }
        ],
    )

    response = client.get(
        "/logs?limit=10&start_date=2026-03-12&end_date=2026-03-13&severity=error"
    )

    assert response.status_code == 200
    payload = response.json()
    assert "logs" in payload
    assert len(payload["logs"]) == 1
    assert payload["logs"][0]["request_id"] == "req-1"


def test_logs_endpoint_invalid_date_range_returns_422(client):
    response = client.get("/logs?start_date=2026-03-14&end_date=2026-03-13")

    assert response.status_code == 422
