import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "metrics" / "search_logs.db"
_DB_LOCK = threading.RLock()


def _resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_logs (
                    request_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    top_k INTEGER NOT NULL,
                    alpha REAL NOT NULL,
                    result_count INTEGER NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()


def log_request(
    query: str,
    latency_ms: float,
    top_k: int,
    alpha: float,
    result_count: int,
    error: Optional[str] = None,
    request_id: Optional[str] = None,
    created_at: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    path = _resolve_db_path(db_path)
    init_db(path)

    request_identifier = request_id or str(uuid.uuid4())
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            connection.execute(
                """
                INSERT INTO search_logs (
                    request_id,
                    query,
                    latency_ms,
                    top_k,
                    alpha,
                    result_count,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_identifier,
                    query,
                    float(latency_ms),
                    int(top_k),
                    float(alpha),
                    int(result_count),
                    error,
                    timestamp,
                ),
            )
            connection.commit()

    return request_identifier


def get_logs(
    limit: int = 100,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    path = _resolve_db_path(db_path)
    init_db(path)

    safe_limit = max(1, min(int(limit), 1000))

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    request_id,
                    query,
                    latency_ms,
                    top_k,
                    alpha,
                    result_count,
                    error,
                    created_at
                FROM search_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

    return [dict(row) for row in rows]
