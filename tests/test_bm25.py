"""Tests for BM25Index."""
import json
import pytest
from backend.app.search.bm25 import BM25Index


@pytest.fixture
def toy_corpus():
    """A 3-document toy corpus for testing."""
    return [
        {
            "doc_id": "doc_python",
            "title": "Python Programming",
            "text": "Python is a versatile programming language used for web development, data science, and machine learning."
        },
        {
            "doc_id": "doc_java",
            "title": "Java Development",
            "text": "Java is an object-oriented programming language popular for enterprise applications and Android development."
        },
        {
            "doc_id": "doc_cooking",
            "title": "Cooking Recipes",
            "text": "Learn how to cook delicious pasta, pizza, and other Italian dishes with fresh ingredients."
        },
    ]


@pytest.fixture
def built_index(toy_corpus, tmp_path):
    """A BM25Index built from the toy corpus, using a temp directory."""
    index = BM25Index(index_dir=str(tmp_path / "bm25"))
    index.build(toy_corpus)
    return index


class TestBM25Build:

    def test_build_sets_bm25(self, built_index):
        assert built_index.bm25 is not None

    def test_build_stores_doc_ids(self, built_index):
        assert built_index.doc_ids == ["doc_python", "doc_java", "doc_cooking"]

    def test_build_empty_raises(self, tmp_path):
        index = BM25Index(index_dir=str(tmp_path / "empty"))
        with pytest.raises(ValueError, match="empty document list"):
            index.build([])


class TestBM25Query:

    def test_python_query_ranks_python_first(self, built_index):
        results = built_index.query("python")
        assert len(results) > 0
        assert results[0]["doc_id"] == "doc_python"

    def test_cooking_query_ranks_cooking_first(self, built_index):
        results = built_index.query("cooking pasta Italian")
        assert len(results) > 0
        assert results[0]["doc_id"] == "doc_cooking"

    def test_top_k_limits_results(self, built_index):
        results = built_index.query("programming", k=1)
        assert len(results) == 1

    def test_top_k_larger_than_corpus(self, built_index):
        results = built_index.query("programming", k=100)
        assert len(results) <= 3

    def test_empty_query_returns_empty(self, built_index):
        results = built_index.query("")
        assert results == []

    def test_result_has_doc_id_and_score(self, built_index):
        results = built_index.query("python")
        for r in results:
            assert "doc_id" in r
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_scores_are_descending(self, built_index):
        results = built_index.query("programming language")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25SaveLoad:

    def test_save_creates_expected_files(self, built_index, tmp_path):
        built_index.save()
        index_dir = tmp_path / "bm25"
        assert (index_dir / "bm25_index.pkl").exists()
        assert (index_dir / "metadata.json").exists()
        assert (index_dir / "documents.jsonl").exists()

    def test_metadata_content(self, built_index, tmp_path):
        built_index.save()
        metadata_path = tmp_path / "bm25" / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert metadata["num_documents"] == 3
        assert metadata["doc_ids"] == ["doc_python", "doc_java", "doc_cooking"]

    def test_load_and_query_matches_original(self, built_index, tmp_path):
        built_index.save()

        loaded_index = BM25Index(index_dir=str(tmp_path / "bm25"))
        loaded_index.load()

        original_results = built_index.query("python", k=3)
        loaded_results = loaded_index.query("python", k=3)

        assert len(original_results) == len(loaded_results)
        for orig, loaded in zip(original_results, loaded_results):
            assert orig["doc_id"] == loaded["doc_id"]
            assert abs(orig["score"] - loaded["score"]) < 1e-6

    def test_load_missing_dir_raises(self, tmp_path):
        index = BM25Index(index_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            index.load()

    def test_query_without_build_raises(self, tmp_path):
        index = BM25Index(index_dir=str(tmp_path / "empty"))
        with pytest.raises(ValueError, match="not built or loaded"):
            index.query("test")
