#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

if __package__ in {None, ""}:
    project_root_for_path = Path(__file__).resolve().parents[1]
    if str(project_root_for_path) not in sys.path:
        sys.path.insert(0, str(project_root_for_path))

from app.search.hybrid import hybrid_search
from backend.app.search.bm25 import BM25Index
from backend.app.search.vector import VectorIndex


ALPHA = 0.5
K_AT = 10
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_CSV = PROJECT_ROOT / "data" / "metrics" / "experiments.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _parse_query_obj(payload: Dict[str, object]) -> Tuple[str, str]:
    query_id = payload.get("query_id") or payload.get("qid") or payload.get("id")
    query_text = payload.get("query") or payload.get("text") or payload.get("question")

    if query_id is None or query_text is None:
        raise ValueError("Each query row must include query_id/id and query/text")

    query_id_str = str(query_id).strip()
    query_text_str = str(query_text).strip()

    if not query_id_str or not query_text_str:
        raise ValueError("Each query row must include non-empty query_id and query text")

    return query_id_str, query_text_str


def load_queries_jsonl(path: Path) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in queries file at line {line_number}: {exc}") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"Query JSONL line {line_number} must be a JSON object")

            queries.append(_parse_query_obj(payload))

    if not queries:
        raise ValueError("No queries found in queries JSONL file")

    return queries


def _coerce_doc_relevance_map(value: object) -> Dict[str, float]:
    relevance_map: Dict[str, float] = {}

    if isinstance(value, dict):
        for doc_id, relevance in value.items():
            try:
                relevance_map[str(doc_id)] = float(relevance)
            except (TypeError, ValueError):
                continue
        return relevance_map

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                relevance_map[item] = 1.0
            elif isinstance(item, dict):
                doc_id = item.get("doc_id") or item.get("id")
                if doc_id is None:
                    continue
                relevance = item.get("relevance", item.get("rel", 1.0))
                try:
                    relevance_map[str(doc_id)] = float(relevance)
                except (TypeError, ValueError):
                    continue
        return relevance_map

    return relevance_map


def load_qrels_json(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "qrels" in payload:
        payload = payload["qrels"]

    qrels: Dict[str, Dict[str, float]] = {}

    if isinstance(payload, dict):
        for query_id, relevance_info in payload.items():
            qrels[str(query_id)] = _coerce_doc_relevance_map(relevance_info)
        return qrels

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue

            query_id = row.get("query_id") or row.get("qid") or row.get("id")
            if query_id is None:
                continue

            query_id_str = str(query_id)

            if "doc_id" in row:
                doc_id = str(row["doc_id"])
                relevance = row.get("relevance", row.get("rel", 1.0))
                try:
                    relevance_val = float(relevance)
                except (TypeError, ValueError):
                    continue
                qrels.setdefault(query_id_str, {})[doc_id] = relevance_val
                continue

            relevance_info = (
                row.get("relevance")
                or row.get("relevances")
                or row.get("qrels")
                or row.get("relevant_docs")
                or row.get("relevant")
            )
            parsed = _coerce_doc_relevance_map(relevance_info)
            if parsed:
                qrels.setdefault(query_id_str, {}).update(parsed)

        return qrels

    raise ValueError("Unsupported qrels format. Expected JSON object or JSON array.")


def recall_at_k(retrieved_doc_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    relevant_doc_ids = {doc_id for doc_id, rel in relevance_map.items() if rel > 0.0}
    if not relevant_doc_ids:
        return 0.0

    retrieved_relevant = sum(1 for doc_id in retrieved_doc_ids[:k] if doc_id in relevant_doc_ids)
    return retrieved_relevant / len(relevant_doc_ids)


def mrr_at_k(retrieved_doc_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if relevance_map.get(doc_id, 0.0) > 0.0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_doc_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    total = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = max(0.0, relevance_map.get(doc_id, 0.0))
        if relevance == 0.0:
            continue
        total += (2.0 ** relevance - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved_doc_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    dcg = dcg_at_k(retrieved_doc_ids, relevance_map, k)

    ideal_relevances = sorted((max(0.0, rel) for rel in relevance_map.values()), reverse=True)[:k]
    if not ideal_relevances:
        return 0.0

    idcg = 0.0
    for rank, relevance in enumerate(ideal_relevances, start=1):
        if relevance == 0.0:
            continue
        idcg += (2.0 ** relevance - 1.0) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_queries(
    queries: List[Tuple[str, str]],
    qrels: Dict[str, Dict[str, float]],
    bm25_index: BM25Index,
    vector_index: VectorIndex,
) -> Dict[str, float]:
    ndcg_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []

    for query_id, query_text in tqdm(queries, desc="Evaluating", unit="query"):
        results = hybrid_search(
            query=query_text,
            bm25_index=bm25_index,
            vector_index=vector_index,
            top_k=K_AT,
            alpha=ALPHA,
            method="weighted",
        )
        retrieved_doc_ids = [row["doc_id"] for row in results]
        relevance_map = qrels.get(query_id, {})

        ndcg_scores.append(ndcg_at_k(retrieved_doc_ids, relevance_map, K_AT))
        recall_scores.append(recall_at_k(retrieved_doc_ids, relevance_map, K_AT))
        mrr_scores.append(mrr_at_k(retrieved_doc_ids, relevance_map, K_AT))

    return {
        "ndcg_at_10": _mean(ndcg_scores),
        "recall_at_10": _mean(recall_scores),
        "mrr_at_10": _mean(mrr_scores),
        "num_queries": float(len(queries)),
    }


def append_experiment_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = [
        "timestamp",
        "git_commit",
        "queries_file",
        "qrels_file",
        "alpha",
        "k",
        "num_queries",
        "ndcg_at_10",
        "recall_at_10",
        "mrr_at_10",
    ]

    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid search and log experiment metrics")
    parser.add_argument("queries_jsonl", type=str, help="Path to queries JSONL file")
    parser.add_argument("qrels_json", type=str, help="Path to qrels JSON file")
    parser.add_argument(
        "--bm25-index-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "index" / "bm25"),
        help="BM25 index directory",
    )
    parser.add_argument(
        "--vector-index-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "index" / "vector"),
        help="Vector index directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_EXPERIMENTS_CSV),
        help="Path to experiments CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    queries_path = Path(args.queries_jsonl)
    qrels_path = Path(args.qrels_json)
    output_csv = Path(args.output_csv)

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries JSONL not found: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels JSON not found: {qrels_path}")

    queries = load_queries_jsonl(queries_path)
    qrels = load_qrels_json(qrels_path)

    bm25_index = BM25Index(index_dir=args.bm25_index_dir)
    vector_index = VectorIndex(index_dir=args.vector_index_dir)
    bm25_index.load()
    vector_index.load()

    metrics = evaluate_queries(
        queries=queries,
        qrels=qrels,
        bm25_index=bm25_index,
        vector_index=vector_index,
    )

    timestamp = _utc_now_iso()
    git_commit = _get_git_commit()

    row = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "queries_file": str(queries_path),
        "qrels_file": str(qrels_path),
        "alpha": ALPHA,
        "k": K_AT,
        "num_queries": int(metrics["num_queries"]),
        "ndcg_at_10": f"{metrics['ndcg_at_10']:.6f}",
        "recall_at_10": f"{metrics['recall_at_10']:.6f}",
        "mrr_at_10": f"{metrics['mrr_at_10']:.6f}",
    }

    append_experiment_row(output_csv, row)

    print("Evaluation complete")
    print(f"queries={int(metrics['num_queries'])}")
    print(f"ndcg@10={metrics['ndcg_at_10']:.6f}")
    print(f"recall@10={metrics['recall_at_10']:.6f}")
    print(f"mrr@10={metrics['mrr_at_10']:.6f}")
    print(f"appended={output_csv}")
    print(f"commit={git_commit}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
