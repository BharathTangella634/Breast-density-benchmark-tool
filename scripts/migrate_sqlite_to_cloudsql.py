"""Migrate evaluation_runs from local SQLite to Cloud SQL (MySQL).

Usage:
    python scripts/migrate_sqlite_to_postgres.py \
        --sqlite-path data/private/evaluation_history.db \
        --database-url mysql://user:pass@host:3306/benchmark_db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from urllib.parse import urlparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate SQLite evaluation data to Cloud SQL (MySQL)")
    parser.add_argument("--sqlite-path", required=True, help="Path to local SQLite database")
    parser.add_argument("--database-url", required=True, help="MySQL connection string (mysql://user:pass@host:3306/db)")
    args = parser.parse_args()

    try:
        import pymysql
    except ImportError:
        print("pymysql not installed. Run: pip install pymysql")
        sys.exit(1)

    sqlite_conn = sqlite3.connect(args.sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    rows = sqlite_conn.execute(
        """
        SELECT model_name, submission_type, source_filename,
               sample_count, macro_f1, accuracy, balanced_accuracy,
               weighted_f1, quadratic_kappa, created_at
        FROM evaluation_runs
        ORDER BY id ASC
        """
    ).fetchall()
    sqlite_conn.close()

    if not rows:
        print("No rows found in SQLite database.")
        return

    print(f"Found {len(rows)} evaluation run(s) in SQLite.")

    parsed = urlparse(args.database_url)
    mysql_conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/"),
    )
    cur = mysql_conn.cursor()

    cur.execute("""
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
    """)

    cur.execute("""
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
    """)

    inserted = 0
    skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM evaluation_runs WHERE model_name = %s AND created_at = %s LIMIT 1",
            (row["model_name"], row["created_at"]),
        )
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute(
            """
            INSERT INTO evaluation_runs (
                model_name, submission_type, source_filename,
                sample_count, macro_f1, accuracy, balanced_accuracy,
                weighted_f1, quadratic_kappa, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["model_name"],
                row["submission_type"],
                row["source_filename"],
                row["sample_count"],
                row["macro_f1"],
                row["accuracy"],
                row["balanced_accuracy"],
                row["weighted_f1"],
                row["quadratic_kappa"],
                row["created_at"],
            ),
        )
        inserted += 1

    mysql_conn.commit()
    cur.close()
    mysql_conn.close()

    print(f"Migration complete: {inserted} inserted, {skipped} skipped (already exist).")


if __name__ == "__main__":
    main()
