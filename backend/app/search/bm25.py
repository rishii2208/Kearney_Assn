"""
BM25 search index using rank-bm25 library.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 index for document search."""
    
    def __init__(self, index_dir: str = "data/index/bm25"):
        """
        Initialize BM25 index.
        
        Args:
            index_dir: Directory to save/load index files
        """
        self.index_dir = Path(index_dir)
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (lowercase words)
        """
        # Simple tokenization: lowercase and split on whitespace/punctuation
        text = text.lower()
        # Replace common punctuation with spaces
        for char in ".,!?;:()[]{}\"'":
            text = text.replace(char, " ")
        tokens = text.split()
        return [token for token in tokens if token]
    
    def build(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of dicts with keys: doc_id, title, text
        """
        if not documents:
            raise ValueError("Cannot build index with empty document list")
        
        self.documents = documents
        self.doc_ids = [doc['doc_id'] for doc in documents]
        
        # Tokenize all documents (combine title and text)
        tokenized_corpus = []
        for doc in documents:
            combined_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = self.tokenize(combined_text)
            tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Built BM25 index with {len(documents)} documents")
    
    def save(self) -> None:
        """Save index to disk."""
        if self.bm25 is None:
            raise ValueError("Cannot save: index not built yet")
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        index_path = self.index_dir / "bm25_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # Save document metadata
        metadata_path = self.index_dir / "metadata.json"
        metadata = {
            'doc_ids': self.doc_ids,
            'num_documents': len(self.documents)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save full documents for retrieval
        docs_path = self.index_dir / "documents.jsonl"
        with open(docs_path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps(doc) + '\n')
        
        print(f"Saved BM25 index to {self.index_dir}")
    
    def load(self) -> None:
        """Load index from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
        
        # Load BM25 index
        index_path = self.index_dir / "bm25_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        # Load metadata
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.doc_ids = metadata['doc_ids']
        
        # Load documents
        docs_path = self.index_dir / "documents.jsonl"
        self.documents = []
        with open(docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.documents.append(json.loads(line))
        
        print(f"Loaded BM25 index with {len(self.documents)} documents")
    
    def query(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query_text: Search query string
            k: Number of top results to return
            
        Returns:
            List of dicts with keys: doc_id, score
        """
        if self.bm25 is None:
            raise ValueError("Index not built or loaded. Call build() or load() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query_text)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'score': float(scores[idx])
                })
        
        return results
