"""
FAISS vector search index using sentence-transformers embeddings.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """FAISS-based vector index with sentence-transformer embeddings."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, index_dir: str = "data/index/vector", model_name: str = DEFAULT_MODEL):
        """
        Initialize VectorIndex.

        Args:
            index_dir: Directory to save/load index files
            model_name: Sentence-transformer model name
        """
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids: List[str] = []
        self.dim: int = 0

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    @staticmethod
    def _corpus_hash(documents: List[Dict[str, Any]]) -> str:
        """Deterministic hash of doc_ids to detect corpus changes."""
        id_str = "|".join(sorted(d["doc_id"] for d in documents))
        return hashlib.sha256(id_str.encode("utf-8")).hexdigest()

    def build(self, documents: List[Dict[str, Any]]) -> None:
        """
        Encode documents and build a FAISS index.

        Args:
            documents: List of dicts with keys: doc_id, title, text
        """
        if not documents:
            raise ValueError("Cannot build index with empty document list")

        model = self._load_model()

        self.doc_ids = [doc["doc_id"] for doc in documents]

        # Combine title and text for encoding
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in documents]

        # Encode to embeddings
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.dim = embeddings.shape[1]

        # Build inner-product index (equivalent to cosine sim on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        print(f"Built vector index: {len(documents)} docs, dim={self.dim}")

    def save(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("Cannot save: index not built yet")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "dim": self.dim,
            "num_documents": len(self.doc_ids),
            "corpus_hash": self._corpus_hash([{"doc_id": d} for d in self.doc_ids]),
            "timestamp": time.time(),
            "doc_ids": self.doc_ids,
        }
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved vector index to {self.index_dir}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")

        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate model name
        saved_model = metadata.get("model_name")
        if saved_model != self.model_name:
            raise ValueError(
                f"Model mismatch: index was built with '{saved_model}' but "
                f"current config expects '{self.model_name}'. "
                f"Please rebuild the index with: VectorIndex(model_name='{self.model_name}').build(docs)"
            )

        # Validate embedding dimension
        saved_dim = metadata.get("dim")
        model = self._load_model()
        expected_dim = model.get_sentence_embedding_dimension()
        if saved_dim != expected_dim:
            raise ValueError(
                f"Dimension mismatch: index has dim={saved_dim} but model "
                f"'{self.model_name}' produces dim={expected_dim}. "
                f"Please rebuild the index with: VectorIndex().build(docs)"
            )

        self.dim = saved_dim
        self.doc_ids = metadata["doc_ids"]

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        print(f"Loaded vector index: {len(self.doc_ids)} docs, dim={self.dim}")

    def query(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents by cosine similarity.

        Args:
            query_text: Search query string
            k: Number of top results to return

        Returns:
            List of dicts with keys: doc_id, score
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build() or load() first.")

        if not query_text.strip():
            return []

        model = self._load_model()

        # Encode and normalize query
        query_emb = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)

        # Clamp k to number of indexed docs
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(score),
            })

        return results
