from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from app import db
from app.evaluation import EvaluationResult


def initialize_history_db() -> None:
    db.create_tables()


def model_name_exists(model_name: str) -> bool:
    with db.get_connection() as conn:
        row = db.fetchone(
            conn,
            "SELECT 1 FROM evaluation_runs WHERE model_name = ? LIMIT 1",
            (model_name,),
        )
    return row is not None


def record_evaluation(
    *,
    result: EvaluationResult,
    submission_type: str,
    source_filename: str,
) -> dict:
    payload = asdict(result)
    created_at = datetime.now(timezone.utc).isoformat()

    with db.get_connection() as conn:
        cursor = db.execute(
            conn,
            """
            INSERT INTO evaluation_runs (
                model_name, submission_type, source_filename,
                sample_count, macro_f1, accuracy, balanced_accuracy,
                weighted_f1, quadratic_kappa, created_at
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
        run_id = db.lastrowid(cursor)

    return {
        "id": run_id,
        "created_at": created_at,
        "submission_type": submission_type,
        "source_filename": source_filename,
    }


def fetch_history() -> list[dict]:
    with db.get_connection() as conn:
        return db.fetchall(
            conn,
            """
            SELECT
                id, model_name, submission_type, source_filename,
                sample_count, macro_f1, accuracy, balanced_accuracy,
                weighted_f1, quadratic_kappa, created_at
            FROM evaluation_runs
            ORDER BY id DESC
            """,
        )


def fetch_leaderboard() -> list[dict]:
    with db.get_connection() as conn:
        return db.fetchall(
            conn,
            """
            SELECT
                e.model_name,
                MAX(e.macro_f1) AS best_macro_f1,
                MAX(e.quadratic_kappa) AS best_quadratic_kappa,
                MAX(e.accuracy) AS best_accuracy,
                COUNT(*) AS total_runs,
                MAX(e.created_at) AS last_run_at,
                (
                    SELECT e2.submission_type
                    FROM evaluation_runs e2
                    WHERE e2.model_name = e.model_name
                    ORDER BY e2.id DESC
                    LIMIT 1
                ) AS submission_type
            FROM evaluation_runs e
            GROUP BY e.model_name
            ORDER BY best_macro_f1 DESC, best_accuracy DESC, e.model_name ASC
            """,
        )
