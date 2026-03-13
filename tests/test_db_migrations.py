import sqlite3

from app.db import get_logs, init_db


def test_init_db_migrates_old_schema_and_preserves_data(tmp_path):
    db_path = tmp_path / "search_logs.db"

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE search_logs (
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
                "req-legacy-1",
                "legacy query",
                12.5,
                5,
                0.4,
                2,
                None,
                "2026-03-13T00:00:00+00:00",
            ),
        )
        connection.commit()

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(search_logs)").fetchall()]
        assert "user_agent" in columns

        applied_versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        assert 1 in applied_versions

        row = connection.execute(
            """
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
            WHERE request_id = ?
            """,
            ("req-legacy-1",),
        ).fetchone()

    assert row is not None
    assert row[0] == "req-legacy-1"
    assert row[1] == "legacy query"
    assert row[2] == 12.5
    assert row[8] is None

    logs = get_logs(limit=10, db_path=db_path)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "req-legacy-1"
    assert logs[0]["query"] == "legacy query"
    assert logs[0]["user_agent"] is None
