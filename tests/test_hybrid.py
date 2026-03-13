"""Tests for hybrid search score fusion."""

import pytest

from app.search.hybrid import hybrid_search


class FixedIndex:
    """Simple index mock returning fixed query results."""

    def __init__(self, results):
        self._results = results

    def query(self, query: str, k: int = 10):
        return self._results[:k]


def _by_doc_id(results):
    return {item["doc_id"]: item for item in results}


def test_alpha_1_uses_bm25_ordering():
    bm25 = FixedIndex(
        [
            {"doc_id": "d1", "score": 100.0},
            {"doc_id": "d2", "score": 50.0},
            {"doc_id": "d3", "score": 10.0},
        ]
    )
    vector = FixedIndex(
        [
            {"doc_id": "d3", "score": 0.9},
            {"doc_id": "d2", "score": 0.5},
            {"doc_id": "d1", "score": 0.1},
        ]
    )

    results = hybrid_search("python", bm25, vector, top_k=3, alpha=1.0)

    assert [item["doc_id"] for item in results] == ["d1", "d2", "d3"]
    for item in results:
        assert item["hybrid_score"] == pytest.approx(item["bm25_score"])


def test_alpha_0_uses_vector_ordering():
    bm25 = FixedIndex(
        [
            {"doc_id": "d1", "score": 100.0},
            {"doc_id": "d2", "score": 50.0},
            {"doc_id": "d3", "score": 10.0},
        ]
    )
    vector = FixedIndex(
        [
            {"doc_id": "d2", "score": 0.9},
            {"doc_id": "d3", "score": 0.5},
            {"doc_id": "d1", "score": 0.1},
        ]
    )

    results = hybrid_search("python", bm25, vector, top_k=3, alpha=0.0)

    assert [item["doc_id"] for item in results] == ["d2", "d3", "d1"]
    for item in results:
        assert item["hybrid_score"] == pytest.approx(item["vector_score"])


def test_alpha_half_blends_min_max_scores_correctly():
    bm25 = FixedIndex(
        [
            {"doc_id": "a", "score": 10.0},
            {"doc_id": "b", "score": 5.0},
            {"doc_id": "c", "score": 0.0},
        ]
    )
    vector = FixedIndex(
        [
            {"doc_id": "a", "score": 0.2},
            {"doc_id": "b", "score": 0.8},
            {"doc_id": "c", "score": 0.1},
        ]
    )

    results = hybrid_search("python", bm25, vector, top_k=3, alpha=0.5)
    rows = _by_doc_id(results)

    assert [item["doc_id"] for item in results] == ["b", "a", "c"]
    assert rows["a"]["bm25_score"] == pytest.approx(1.0)
    assert rows["a"]["vector_score"] == pytest.approx((0.2 - 0.1) / (0.8 - 0.1))
    assert rows["a"]["hybrid_score"] == pytest.approx(0.5 * 1.0 + 0.5 * ((0.2 - 0.1) / 0.7))

    assert rows["b"]["bm25_score"] == pytest.approx(0.5)
    assert rows["b"]["vector_score"] == pytest.approx(1.0)
    assert rows["b"]["hybrid_score"] == pytest.approx(0.75)

    assert rows["c"]["bm25_score"] == pytest.approx(0.0)
    assert rows["c"]["vector_score"] == pytest.approx(0.0)
    assert rows["c"]["hybrid_score"] == pytest.approx(0.0)


def test_all_equal_bm25_scores_do_not_crash():
    bm25 = FixedIndex(
        [
            {"doc_id": "d1", "score": 2.0},
            {"doc_id": "d2", "score": 2.0},
            {"doc_id": "d3", "score": 2.0},
        ]
    )
    vector = FixedIndex(
        [
            {"doc_id": "d1", "score": 0.1},
            {"doc_id": "d2", "score": 0.3},
            {"doc_id": "d3", "score": 0.2},
        ]
    )

    results = hybrid_search("python", bm25, vector, top_k=3, alpha=0.5)

    assert len(results) == 3
    for item in results:
        assert item["bm25_score"] == pytest.approx(1.0)


def test_doc_missing_from_one_index_does_not_raise_keyerror():
    bm25 = FixedIndex(
        [
            {"doc_id": "a", "score": 3.0},
            {"doc_id": "b", "score": 1.0},
        ]
    )
    vector = FixedIndex(
        [
            {"doc_id": "b", "score": 0.9},
            {"doc_id": "c", "score": 0.8},
        ]
    )

    results = hybrid_search("python", bm25, vector, top_k=3, alpha=0.5)
    rows = _by_doc_id(results)

    assert set(rows.keys()) == {"a", "b", "c"}
    assert rows["a"]["vector_score"] == pytest.approx(0.0)
    assert rows["c"]["bm25_score"] == pytest.approx(0.0)
