import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "metrics" / "search_logs.db"
_DB_LOCK = threading.RLock()
_MIGRATIONS_TABLE = "schema_migrations"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _ensure_search_logs_base_table(connection: sqlite3.Connection) -> None:
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


def _ensure_migrations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_MIGRATIONS_TABLE} (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _get_applied_versions(connection: sqlite3.Connection) -> Set[int]:
    rows = connection.execute(
        f"SELECT version FROM {_MIGRATIONS_TABLE} ORDER BY version"
    ).fetchall()
    return {int(row[0]) for row in rows}


def _has_column(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in columns)


def _migration_1_add_user_agent(connection: sqlite3.Connection) -> None:
    if _has_column(connection, "search_logs", "user_agent"):
        return
    connection.execute("ALTER TABLE search_logs ADD COLUMN user_agent TEXT")


_MIGRATIONS: Dict[int, Callable[[sqlite3.Connection], None]] = {
    1: _migration_1_add_user_agent,
}


def _run_migrations(connection: sqlite3.Connection) -> None:
    _ensure_migrations_table(connection)
    applied_versions = _get_applied_versions(connection)

    for version, migration in sorted(_MIGRATIONS.items()):
        if version in applied_versions:
            continue
        migration(connection)
        connection.execute(
            f"INSERT INTO {_MIGRATIONS_TABLE} (version, applied_at) VALUES (?, ?)",
            (version, _utc_now_iso()),
        )


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            _ensure_search_logs_base_table(connection)
            _run_migrations(connection)
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
    user_agent: Optional[str] = None,
) -> str:
    path = _resolve_db_path(db_path)
    init_db(path)

    request_identifier = request_id or str(uuid.uuid4())
    timestamp = created_at or _utc_now_iso()

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
                    created_at,
                    user_agent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    user_agent,
                ),
            )
            connection.commit()

    return request_identifier


def get_logs_filtered(
    limit: int = 100,
    start_created_at: Optional[str] = None,
    end_created_at: Optional[str] = None,
    severity: str = "all",
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    path = _resolve_db_path(db_path)
    init_db(path)

    safe_limit = max(1, min(int(limit), 1000))
    severity_filter = (severity or "all").lower()

    if severity_filter not in {"all", "error", "success"}:
        raise ValueError("severity must be one of: all, error, success")

    where_clauses: List[str] = []
    params: List[Any] = []

    if start_created_at:
        where_clauses.append("created_at >= ?")
        params.append(start_created_at)

    if end_created_at:
        where_clauses.append("created_at <= ?")
        params.append(end_created_at)

    if severity_filter == "error":
        where_clauses.append("error IS NOT NULL AND TRIM(error) <> ''")
    elif severity_filter == "success":
        where_clauses.append("(error IS NULL OR TRIM(error) = '')")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT
                    request_id,
                    query,
                    latency_ms,
                    top_k,
                    alpha,
                    result_count,
                    error,
                    created_at,
                    user_agent
                FROM search_logs
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*params, safe_limit),
            ).fetchall()

    return [dict(row) for row in rows]


def get_logs(
    limit: int = 100,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    return get_logs_filtered(limit=limit, db_path=db_path)


def _percentile(sorted_values: List[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * quantile
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = position - lower_idx

    lower_val = float(sorted_values[lower_idx])
    upper_val = float(sorted_values[upper_idx])
    return lower_val + (upper_val - lower_val) * weight


def get_metrics(
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[int, float]]:
    path = _resolve_db_path(db_path)
    init_db(path)

    with _DB_LOCK:
        with sqlite3.connect(path, timeout=30, check_same_thread=False) as connection:
            total_requests = int(
                connection.execute("SELECT COUNT(*) FROM search_logs").fetchone()[0]
            )
            total_errors = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM search_logs
                    WHERE error IS NOT NULL AND TRIM(error) <> ''
                    """
                ).fetchone()[0]
            )
            zero_result_queries = int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_logs WHERE result_count = 0"
                ).fetchone()[0]
            )
            latency_rows = connection.execute(
                "SELECT latency_ms FROM search_logs WHERE latency_ms IS NOT NULL ORDER BY latency_ms"
            ).fetchall()

    latencies = [float(row[0]) for row in latency_rows]

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "zero_result_query_count": zero_result_queries,
    }


def get_top_queries(
    limit: int = 10,
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
                    query,
                    COUNT(*) AS count,
                    MAX(created_at) AS last_seen
                FROM search_logs
                GROUP BY query
                ORDER BY count DESC, last_seen DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

    return [dict(row) for row in rows]


def get_zero_result_queries(
    limit: int = 10,
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
                    query,
                    COUNT(*) AS count,
                    MAX(created_at) AS last_seen
                FROM search_logs
                WHERE result_count = 0
                GROUP BY query
                ORDER BY count DESC, last_seen DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

    return [dict(row) for row in rows]
