from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import db

logger = logging.getLogger(__name__)

JOB_METADATA_TTL = 3600


@dataclass
class OnnxJob:
    job_id: str
    model_name: str
    filename: str
    model_path: Path
    status: str = "queued"
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None


class OnnxQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, OnnxJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._inference_times: list[float] = []
        self._evaluate_fn: Any = None
        self._running_job_id: str | None = None
        self._timeout_seconds: int = 3600

    def configure(
        self,
        *,
        evaluate_fn: Any,
        timeout_seconds: int = 3600,
    ) -> None:
        self._evaluate_fn = evaluate_fn
        self._timeout_seconds = timeout_seconds

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._restore_pending_jobs()
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("ONNX queue worker started")

    async def submit(
        self,
        *,
        model_name: str,
        model_path: Path,
        filename: str,
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = OnnxJob(
            job_id=job_id,
            model_name=model_name,
            filename=filename,
            model_path=model_path,
        )
        self._jobs[job_id] = job
        self._persist_job(job)
        await self._queue.put(job_id)

        position = self._queue_position(job_id)
        logger.info(
            "Job %s queued at position %d for model '%s'",
            job_id, position, model_name,
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "queue_position": position,
            "estimated_wait_seconds": self._estimate_wait(position),
        }

    def get_status(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return self._load_job_from_db(job_id)

        position = self._queue_position(job_id) if job.status == "queued" else 0

        response: dict = {
            "job_id": job.job_id,
            "status": job.status,
            "model_name": job.model_name,
            "filename": job.filename,
            "created_at": job.created_at,
        }

        if job.status == "queued":
            response["queue_position"] = position
            response["estimated_wait_seconds"] = self._estimate_wait(position)
        elif job.status == "running":
            response["started_at"] = job.started_at
            elapsed = time.time() - job.started_at if job.started_at else 0
            response["elapsed_seconds"] = round(elapsed, 1)
        elif job.status == "completed":
            response["result"] = job.result
            response["completed_at"] = job.completed_at
        elif job.status == "failed":
            response["error"] = job.error
            response["completed_at"] = job.completed_at

        return response

    def get_queue_info(self) -> dict:
        queued_count = sum(1 for j in self._jobs.values() if j.status == "queued")
        running_job = None
        if self._running_job_id and self._running_job_id in self._jobs:
            rj = self._jobs[self._running_job_id]
            elapsed = time.time() - rj.started_at if rj.started_at else 0
            running_job = {
                "job_id": rj.job_id,
                "model_name": rj.model_name,
                "elapsed_seconds": round(elapsed, 1),
            }

        return {
            "queued": queued_count,
            "running": running_job,
            "avg_inference_seconds": (
                round(self._avg_inference_time(), 1)
                if self._inference_times
                else None
            ),
        }

    # ── Worker ──

    async def _worker(self) -> None:
        logger.info("ONNX queue worker running")
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue

            job.status = "running"
            job.started_at = time.time()
            self._running_job_id = job_id
            self._update_job_db(job_id, status="running", started_at=job.started_at)
            logger.info("Job %s started: model='%s'", job_id, job.model_name)

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._evaluate_fn,
                        model_name=job.model_name,
                        model_path=job.model_path,
                        filename=job.filename,
                    ),
                    timeout=self._timeout_seconds,
                )
                job.status = "completed"
                job.result = result
                job.completed_at = time.time()
                elapsed = job.completed_at - job.started_at
                self._inference_times.append(elapsed)
                if len(self._inference_times) > 20:
                    self._inference_times = self._inference_times[-20:]
                self._update_job_db(
                    job_id, status="completed", completed_at=job.completed_at,
                )
                logger.info("Job %s completed in %.1f s", job_id, elapsed)

            except asyncio.TimeoutError:
                job.status = "failed"
                job.error = (
                    f"Model evaluation timed out after {self._timeout_seconds}s. "
                    "Your model may be too complex for CPU-only inference."
                )
                job.completed_at = time.time()
                self._update_job_db(
                    job_id, status="failed", error=job.error,
                    completed_at=job.completed_at,
                )
                logger.error("Job %s timed out after %ds", job_id, self._timeout_seconds)

            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                self._update_job_db(
                    job_id, status="failed", error=job.error,
                    completed_at=job.completed_at,
                )
                logger.error("Job %s failed: %s", job_id, exc)

            finally:
                self._running_job_id = None
                self._queue.task_done()
                self._delete_model_file(job.model_path)
                self._cleanup_old_jobs()

    # ── Disk cleanup ──

    def _delete_model_file(self, model_path: Path) -> None:
        try:
            if model_path.exists():
                model_path.unlink()
                logger.info("Deleted model file: %s", model_path)
        except Exception as exc:
            logger.warning("Could not delete model file %s: %s", model_path, exc)

    def _cleanup_old_jobs(self) -> None:
        now = time.time()
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.status in ("completed", "failed")
            and job.completed_at
            and (now - job.completed_at) > JOB_METADATA_TTL
        ]
        for jid in expired:
            del self._jobs[jid]
            self._delete_job_db(jid)
        if expired:
            logger.info("Cleaned up %d old job(s)", len(expired))

    # ── Database persistence ──

    def _persist_job(self, job: OnnxJob) -> None:
        with db.get_connection() as conn:
            if db.is_cloud_db():
                db.execute(
                    conn,
                    """
                    INSERT INTO onnx_jobs
                        (job_id, model_name, filename, model_path, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON DUPLICATE KEY UPDATE
                        status = VALUES(status),
                        model_name = VALUES(model_name)
                    """,
                    (
                        job.job_id, job.model_name, job.filename,
                        str(job.model_path), job.status, job.created_at,
                    ),
                )
            else:
                db.execute(
                    conn,
                    """
                    INSERT OR REPLACE INTO onnx_jobs
                        (job_id, model_name, filename, model_path, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id, job.model_name, job.filename,
                        str(job.model_path), job.status, job.created_at,
                    ),
                )

    def _update_job_db(
        self, job_id: str, *,
        status: str,
        started_at: float | None = None,
        completed_at: float | None = None,
        error: str | None = None,
    ) -> None:
        with db.get_connection() as conn:
            db.execute(
                conn,
                """
                UPDATE onnx_jobs
                SET status = ?, started_at = COALESCE(?, started_at),
                    completed_at = COALESCE(?, completed_at),
                    error = COALESCE(?, error)
                WHERE job_id = ?
                """,
                (status, started_at, completed_at, error, job_id),
            )

    def _delete_job_db(self, job_id: str) -> None:
        with db.get_connection() as conn:
            db.execute(conn, "DELETE FROM onnx_jobs WHERE job_id = ?", (job_id,))

    def _load_job_from_db(self, job_id: str) -> dict | None:
        with db.get_connection() as conn:
            row = db.fetchone(
                conn,
                "SELECT * FROM onnx_jobs WHERE job_id = ?",
                (job_id,),
            )
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "model_name": row["model_name"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "error": row["error"],
            "completed_at": row["completed_at"],
        }

    def _restore_pending_jobs(self) -> None:
        with db.get_connection() as conn:
            db.execute(
                conn,
                "UPDATE onnx_jobs SET status = 'queued', started_at = NULL "
                "WHERE status = 'running'",
            )
            rows = db.fetchall(
                conn,
                "SELECT * FROM onnx_jobs WHERE status = 'queued' ORDER BY created_at",
            )

        restored = 0
        for row in rows:
            model_path = Path(row["model_path"])
            if not model_path.exists():
                logger.warning(
                    "Skipping restored job %s — model file missing: %s",
                    row["job_id"], model_path,
                )
                with db.get_connection() as conn:
                    db.execute(
                        conn,
                        "UPDATE onnx_jobs SET status = 'failed', "
                        "error = 'Model file missing after server restart' "
                        "WHERE job_id = ?",
                        (row["job_id"],),
                    )
                continue

            job = OnnxJob(
                job_id=row["job_id"],
                model_name=row["model_name"],
                filename=row["filename"],
                model_path=model_path,
                status="queued",
                created_at=row["created_at"],
            )
            self._jobs[job.job_id] = job
            self._queue.put_nowait(job.job_id)
            restored += 1

        if restored:
            logger.info("Restored %d pending job(s) from database", restored)

    # ── Helpers ──

    def _queue_position(self, job_id: str) -> int:
        position = 0
        for jid, job in self._jobs.items():
            if job.status == "queued":
                position += 1
                if jid == job_id:
                    return position
        return 0

    def _avg_inference_time(self) -> float:
        if not self._inference_times:
            return 600.0
        return sum(self._inference_times) / len(self._inference_times)

    def _estimate_wait(self, position: int) -> int:
        if position <= 0:
            return 0
        avg = self._avg_inference_time()
        running_remaining = 0
        if self._running_job_id and self._running_job_id in self._jobs:
            rj = self._jobs[self._running_job_id]
            if rj.started_at:
                elapsed = time.time() - rj.started_at
                running_remaining = max(0, avg - elapsed)
            return int(running_remaining + (position - 1) * avg)
        return int(position * avg)


onnx_queue = OnnxQueue()
