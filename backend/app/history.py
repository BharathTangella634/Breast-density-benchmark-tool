from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation import EvaluationResult


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_db(db_path: Path) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
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
                quadratic_kappa REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def record_evaluation(
    *,
    db_path: Path,
    result: EvaluationResult,
    submission_type: str,
    source_filename: str,
) -> dict:
    payload = asdict(result)
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs (
                model_name,
                submission_type,
                source_filename,
                sample_count,
                macro_f1,
                accuracy,
                balanced_accuracy,
                weighted_f1,
                quadratic_kappa,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["model_name"],
                submission_type,
                source_filename,
                payload["sample_count"],
                payload["macro_f1"],
                payload["accuracy"],
                payload["balanced_accuracy"],
                payload["weighted_f1"],
                payload["quadratic_kappa"],
                created_at,
            ),
        )
        run_id = cursor.lastrowid

    return {
        "id": run_id,
        "created_at": created_at,
        "submission_type": submission_type,
        "source_filename": source_filename,
    }


def fetch_history(db_path: Path) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                model_name,
                submission_type,
                source_filename,
                sample_count,
                macro_f1,
                accuracy,
                balanced_accuracy,
                weighted_f1,
                quadratic_kappa,
                created_at
            FROM evaluation_runs
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_leaderboard(db_path: Path) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                model_name,
                MAX(macro_f1) AS best_macro_f1,
                MAX(accuracy) AS best_accuracy,
                COUNT(*) AS total_runs,
                MAX(created_at) AS last_run_at
            FROM evaluation_runs
            GROUP BY model_name
            ORDER BY best_macro_f1 DESC, best_accuracy DESC, model_name ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]
