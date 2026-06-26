from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from app.config import settings


def is_cloud_db() -> bool:
    return bool(settings.database_url)


def _parse_mysql_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/"),
    }


@contextmanager
def get_connection():
    if is_cloud_db():
        import pymysql

        params = _parse_mysql_url(settings.database_url)
        conn = pymysql.connect(**params, autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        db_path = settings.history_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _adapt_sql(sql: str) -> str:
    if is_cloud_db():
        return sql.replace("?", "%s")
    return sql


def execute(conn: Any, sql: str, params: tuple = ()) -> Any:
    cursor = conn.cursor()
    cursor.execute(_adapt_sql(sql), params)
    return cursor


def fetchall(conn: Any, sql: str, params: tuple = ()) -> list[dict]:
    cursor = execute(conn, sql, params)
    if cursor.description is None:
        return []
    if is_cloud_db():
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    return [dict(row) for row in cursor.fetchall()]


def fetchone(conn: Any, sql: str, params: tuple = ()) -> dict | None:
    cursor = execute(conn, sql, params)
    row = cursor.fetchone()
    if row is None:
        return None
    if is_cloud_db():
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return dict(row)


def lastrowid(cursor: Any) -> int | None:
    return cursor.lastrowid


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    submission_type TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    macro_f1 REAL NOT NULL,
    accuracy REAL NOT NULL,
    balanced_accuracy REAL NOT NULL,
    weighted_f1 REAL NOT NULL,
    macro_precision REAL,
    macro_recall REAL,
    quadratic_kappa REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS onnx_jobs (
    job_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    model_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL
);
"""

_MYSQL_EVALUATION_RUNS = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    submission_type VARCHAR(50) NOT NULL,
    source_filename VARCHAR(500) NOT NULL,
    sample_count INT NOT NULL,
    macro_f1 DOUBLE NOT NULL,
    accuracy DOUBLE NOT NULL,
    balanced_accuracy DOUBLE NOT NULL,
    weighted_f1 DOUBLE NOT NULL,
    macro_precision DOUBLE,
    macro_recall DOUBLE,
    quadratic_kappa DOUBLE,
    created_at VARCHAR(50) NOT NULL
)
"""

_MYSQL_ONNX_JOBS = """
CREATE TABLE IF NOT EXISTS onnx_jobs (
    job_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    model_path VARCHAR(1000) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    error TEXT,
    created_at DOUBLE NOT NULL,
    started_at DOUBLE,
    completed_at DOUBLE
)
"""


def create_tables() -> None:
    with get_connection() as conn:
        if is_cloud_db():
            cursor = conn.cursor()
            cursor.execute(_MYSQL_EVALUATION_RUNS)
            cursor.execute(_MYSQL_ONNX_JOBS)
        else:
            conn.executescript(_SQLITE_SCHEMA)
            existing_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(evaluation_runs)"
                ).fetchall()
            }
            if "macro_precision" not in existing_columns:
                conn.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN macro_precision REAL"
                )
            if "macro_recall" not in existing_columns:
                conn.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN macro_recall REAL"
                )
