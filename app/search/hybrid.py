"""Hybrid search fusion helpers for BM25 and vector retrieval."""

from typing import Any, Dict, List, Literal, Optional


FusionMethod = Literal["weighted", "rrf"]


def _to_score_map(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Convert [{doc_id, score}] results into a doc_id -> score mapping."""
    return {item["doc_id"]: float(item["score"]) for item in results}


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}

    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {
            doc_id: (1.0 if score > 0.0 else 0.0)
            for doc_id, score in scores.items()
        }

    scale = maximum - minimum
    return {doc_id: (score - minimum) / scale for doc_id, score in scores.items()}


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[Dict[str, float]]:
    """
    Combine two ranked result lists using Reciprocal Rank Fusion (RRF).

    `bm25_score` and `vector_score` in the output are the per-source RRF contributions.
    """
    if top_k <= 0:
        return []

    bm25_ranks = {item["doc_id"]: rank for rank, item in enumerate(bm25_results, start=1)}
    vector_ranks = {item["doc_id"]: rank for rank, item in enumerate(vector_results, start=1)}

    merged_doc_ids = set(bm25_ranks) | set(vector_ranks)

    fused: List[Dict[str, float]] = []
    for doc_id in merged_doc_ids:
        bm25_component = 1.0 / (rrf_k + bm25_ranks[doc_id]) if doc_id in bm25_ranks else 0.0
        vector_component = 1.0 / (rrf_k + vector_ranks[doc_id]) if doc_id in vector_ranks else 0.0
        fused.append(
            {
                "doc_id": doc_id,
                "bm25_score": bm25_component,
                "vector_score": vector_component,
                "hybrid_score": bm25_component + vector_component,
            }
        )

    fused.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return fused[:top_k]


def hybrid_search(
    query: str,
    bm25_index: Any,
    vector_index: Any,
    top_k: int = 10,
    alpha: float = 0.5,
    method: FusionMethod = "weighted",
    fetch_k: Optional[int] = None,
    rrf_k: int = 60,
) -> List[Dict[str, float]]:
    """
    Run hybrid retrieval for a query.

    For `method="weighted"`, source scores are min-max normalized before fusion:
      hybrid_score = alpha * bm25_score + (1 - alpha) * vector_score

    For `method="rrf"`, reciprocal rank fusion is used as an alternative.

    Returns:
        List[dict] with keys: doc_id, bm25_score, vector_score, hybrid_score
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    if method not in {"weighted", "rrf"}:
        raise ValueError("method must be either 'weighted' or 'rrf'")

    candidate_k = fetch_k if fetch_k is not None else top_k

    bm25_results = bm25_index.query(query, k=candidate_k)
    vector_results = vector_index.query(query, k=candidate_k)

    if method == "rrf":
        return reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    bm25_norm = _min_max_normalize(_to_score_map(bm25_results))
    vector_norm = _min_max_normalize(_to_score_map(vector_results))

    merged_doc_ids = set(bm25_norm) | set(vector_norm)

    fused: List[Dict[str, float]] = []
    for doc_id in merged_doc_ids:
        bm25_score = bm25_norm.get(doc_id, 0.0)
        vector_score = vector_norm.get(doc_id, 0.0)
        hybrid_score = alpha * bm25_score + (1.0 - alpha) * vector_score
        fused.append(
            {
                "doc_id": doc_id,
                "bm25_score": bm25_score,
                "vector_score": vector_score,
                "hybrid_score": hybrid_score,
            }
        )

    fused.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return fused[:top_k]
