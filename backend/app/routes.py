import csv
import json
import logging
import re
import time
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.db import (
    get_logs_filtered,
    get_metrics,
    get_top_queries,
    get_zero_result_queries,
    log_request,
)
from app.search.hybrid import hybrid_search


logger = logging.getLogger(__name__)
router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_CSV_PATH = PROJECT_ROOT / "data" / "metrics" / "experiments.csv"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


def get_bm25_index(request: Request):
    return getattr(request.app.state, "bm25_index", None)


def get_vector_index(request: Request):
    return getattr(request.app.state, "vector_index", None)


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


def _render_prometheus_metrics(metrics: Dict[str, float]) -> str:
    lines = [
        "# HELP search_requests_total Total number of search requests",
        "# TYPE search_requests_total counter",
        f"search_requests_total {int(metrics['total_requests'])}",
        "# HELP search_errors_total Total number of search requests with an error",
        "# TYPE search_errors_total counter",
        f"search_errors_total {int(metrics['total_errors'])}",
        "# HELP search_latency_p50_ms 50th percentile search latency in milliseconds",
        "# TYPE search_latency_p50_ms gauge",
        f"search_latency_p50_ms {float(metrics['latency_p50_ms']):.6f}",
        "# HELP search_latency_p95_ms 95th percentile search latency in milliseconds",
        "# TYPE search_latency_p95_ms gauge",
        f"search_latency_p95_ms {float(metrics['latency_p95_ms']):.6f}",
        "# HELP search_zero_result_queries_total Number of zero-result search queries",
        "# TYPE search_zero_result_queries_total counter",
        f"search_zero_result_queries_total {int(metrics['zero_result_query_count'])}",
    ]
    return "\n".join(lines) + "\n"


def _parse_experiment_row(row: Dict[str, str]) -> Dict[str, Any]:
    parsed = dict(row)

    float_fields = ["alpha", "ndcg_at_10", "recall_at_10", "mrr_at_10"]
    int_fields = ["k", "num_queries"]

    for field in float_fields:
        value = parsed.get(field)
        if value in (None, ""):
            continue
        try:
            parsed[field] = float(value)
        except (TypeError, ValueError):
            pass

    for field in int_fields:
        value = parsed.get(field)
        if value in (None, ""):
            continue
        try:
            parsed[field] = int(float(value))
        except (TypeError, ValueError):
            pass

    return parsed


def _read_experiments_csv() -> List[Dict[str, Any]]:
    if not EXPERIMENTS_CSV_PATH.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(EXPERIMENTS_CSV_PATH, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row:
                continue
            rows.append(_parse_experiment_row(row))

    return rows


def _to_utc_day_bounds(
    start_date: date | None,
    end_date: date | None,
) -> tuple[str | None, str | None]:
    start_iso = None
    end_iso = None

    if start_date is not None:
        start_iso = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc).isoformat()

    if end_date is not None:
        end_iso = datetime.combine(end_date, dt_time.max, tzinfo=timezone.utc).isoformat()

    return start_iso, end_iso


@router.post("/search")
async def search(
    payload: SearchRequest,
    request: Request,
    bm25_index: Any = Depends(get_bm25_index),
    vector_index: Any = Depends(get_vector_index),
) -> Dict[str, List[dict]]:
    started_at = time.perf_counter()
    result_count = 0
    error_message = None

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
                    "title": doc.get("title") or item["doc_id"],
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


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    try:
        metrics_payload = get_metrics()
    except Exception:
        logger.exception("Failed to read metrics from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read metrics")

    return PlainTextResponse(
        content=_render_prometheus_metrics(metrics_payload),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/experiments")
async def experiments() -> Dict[str, List[dict]]:
    try:
        return {"experiments": _read_experiments_csv()}
    except Exception:
        logger.exception("Failed to read experiments.csv")
        raise HTTPException(status_code=500, detail="Failed to read experiments")


@router.get("/logs")
async def logs(
    limit: int = Query(default=100, ge=1, le=1000),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    severity: str = Query(default="all", pattern="^(all|error|success)$"),
) -> Dict[str, List[dict]]:
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")

    start_created_at, end_created_at = _to_utc_day_bounds(start_date, end_date)

    try:
        rows = get_logs_filtered(
            limit=limit,
            start_created_at=start_created_at,
            end_created_at=end_created_at,
            severity=severity,
        )
        return {"logs": rows}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to read request logs from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read logs")


@router.get("/metrics/top-queries")
async def top_queries(limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, List[dict]]:
    try:
        return {"top_queries": get_top_queries(limit=limit)}
    except Exception:
        logger.exception("Failed to read top queries from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read top queries")


@router.get("/metrics/zero-result-queries")
async def zero_result_queries(
    limit: int = Query(default=10, ge=1, le=100),
) -> Dict[str, List[dict]]:
    try:
        return {"zero_result_queries": get_zero_result_queries(limit=limit)}
    except Exception:
        logger.exception("Failed to read zero-result queries from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read zero-result queries")


@router.get("/top-queries")
async def top_queries_v1(limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, List[dict]]:
    try:
        return {"top_queries": get_top_queries(limit=limit)}
    except Exception:
        logger.exception("Failed to read top queries from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read top queries")


@router.get("/zero-result-queries")
async def zero_result_queries_v1(
    limit: int = Query(default=10, ge=1, le=100),
) -> Dict[str, List[dict]]:
    try:
        return {"zero_result_queries": get_zero_result_queries(limit=limit)}
    except Exception:
        logger.exception("Failed to read zero-result queries from sqlite")
        raise HTTPException(status_code=500, detail="Failed to read zero-result queries")
