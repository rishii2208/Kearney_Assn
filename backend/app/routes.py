import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.db import log_request
from app.search.hybrid import hybrid_search


logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


def _load_documents_from_jsonl(jsonl_path: Path) -> Dict[str, dict]:
    documents: Dict[str, dict] = {}

    if not jsonl_path.exists():
        logger.warning("Document JSONL not found: %s", jsonl_path)
        return documents

    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                doc_id = payload.get("doc_id")
                if doc_id:
                    documents[doc_id] = payload
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL row in %s", jsonl_path)

    return documents


def _build_snippet(text: str, query: str, snippet_len: int = 200) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    if not cleaned:
        return ""

    if len(cleaned) <= snippet_len:
        return cleaned

    keywords = re.findall(r"\w+", query.lower())
    lower_text = cleaned.lower()

    match_index = -1
    for keyword in keywords:
        match_index = lower_text.find(keyword)
        if match_index != -1:
            break

    if match_index == -1:
        return cleaned[:snippet_len]

    half = snippet_len // 2
    start = max(0, match_index - half)
    end = min(len(cleaned), start + snippet_len)
    start = max(0, end - snippet_len)
    return cleaned[start:end]


@router.post("/search")
async def search(payload: SearchRequest, request: Request) -> Dict[str, List[dict]]:
    started_at = time.perf_counter()
    result_count = 0
    error_message = None

    bm25_index = getattr(request.app.state, "bm25_index", None)
    vector_index = getattr(request.app.state, "vector_index", None)

    try:
        if bm25_index is None or vector_index is None:
            raise HTTPException(status_code=503, detail="Search indexes are not loaded")

        results = hybrid_search(
            query=payload.query,
            bm25_index=bm25_index,
            vector_index=vector_index,
            top_k=payload.top_k,
            alpha=payload.alpha,
            method="weighted",
        )

        docs_jsonl = bm25_index.index_dir / "documents.jsonl"
        documents_by_id = _load_documents_from_jsonl(docs_jsonl)

        response_results = []
        for item in results:
            doc = documents_by_id.get(item["doc_id"], {})
            snippet = _build_snippet(doc.get("text", ""), payload.query)
            response_results.append(
                {
                    "doc_id": item["doc_id"],
                    "bm25_score": item["bm25_score"],
                    "vector_score": item["vector_score"],
                    "hybrid_score": item["hybrid_score"],
                    "snippet": snippet,
                }
            )

        result_count = len(response_results)
        return {"results": response_results}
    except HTTPException as exc:
        error_message = str(exc.detail)
        raise
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Unhandled search failure")
        raise
    finally:
        latency_ms = (time.perf_counter() - started_at) * 1000
        try:
            log_request(
                query=payload.query,
                latency_ms=latency_ms,
                top_k=payload.top_k,
                alpha=payload.alpha,
                result_count=result_count,
                error=error_message,
            )
        except Exception as exc:
            logger.warning("Failed to log search request to sqlite: %s", exc)
