import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from app.config import settings
from app.evaluation import EvaluationResult, evaluate_predictions
from app.history import (
    fetch_history,
    fetch_leaderboard,
    initialize_history_db,
    model_name_exists,
    record_evaluation,
)
from app.onnx_inference import run_onnx_benchmark
from app.queue import onnx_queue


def _max_bytes(megabytes: int) -> int:
    return megabytes * 1024 * 1024


def _validate_upload_size(upload: UploadFile, *, max_mb: int) -> None:
    size = getattr(upload, "size", None)
    if size is not None and size > _max_bytes(max_mb):
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file is too large. Maximum allowed size is {max_mb} MB.",
        )


def _check_model_name_unique(model_name: str) -> None:
    if model_name_exists(model_name):
        raise HTTPException(
            status_code=409,
            detail=(
                f"A model named '{model_name}' already exists on the leaderboard. "
                "Please choose a different model name."
            ),
        )


def _evaluation_response(*, result: EvaluationResult, saved_run: dict) -> dict:
    return {
        "run": saved_run,
        "model_name": result.model_name,
        "sample_count": result.sample_count,
        "primary_metric": {"name": "macro_f1", "value": result.macro_f1},
        "secondary_metrics": {
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "weighted_f1": result.weighted_f1,
            "quadratic_kappa": result.quadratic_kappa,
        },
        "per_class_f1": result.per_class_f1,
    }


def _run_onnx_job(*, model_name: str, model_path: Path, filename: str) -> dict:
    """Execute ONNX evaluation synchronously (called by queue worker thread)."""
    tmp_dir = tempfile.mkdtemp(prefix="onnx_eval_")
    work_model_path = Path(tmp_dir) / filename
    shutil.copy2(model_path, work_model_path)

    try:
        try:
            import onnx
            model_proto = onnx.load(str(work_model_path), load_external_data=False)
            has_external = any(
                tensor.HasField("data_location")
                and tensor.data_location == onnx.TensorProto.EXTERNAL
                for tensor in model_proto.graph.initializer
            )
            if has_external:
                try:
                    model_proto = onnx.load(str(work_model_path))
                except Exception:
                    raise ValueError(
                        "This ONNX model requires an external .data file which "
                        "cannot be uploaded separately. Please re-export with "
                        "embedded weights. PyTorch: torch.onnx.export(model, ..., "
                        "'model.onnx') without external_data threshold. "
                        "ONNX: onnx.save(model, 'model.onnx', "
                        "save_as_external_data=False)."
                    )
                selfcontained_path = Path(tmp_dir) / "model_selfcontained.onnx"
                onnx.save(model_proto, str(selfcontained_path), save_as_external_data=False)
                work_model_path = selfcontained_path
        except ValueError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).debug("ONNX pre-check skipped: %s", exc)

        predictions_csv = run_onnx_benchmark(
            model_path=work_model_path,
            image_manifest_csv=settings.image_manifest_csv,
            image_root=settings.image_root,
            input_size=settings.onnx_input_size,
            input_channels=settings.onnx_input_channels,
            allowed_labels=settings.allowed_labels,
        )
        result = evaluate_predictions(
            predictions_csv=predictions_csv,
            ground_truth_csv=settings.ground_truth_csv,
            model_name=model_name,
            allowed_labels=settings.allowed_labels,
            include_quadratic_kappa=settings.enable_quadratic_kappa,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    saved_run = record_evaluation(
        result=result,
        submission_type="onnx_model",
        source_filename=filename,
    )

    return _evaluation_response(result=result, saved_run=saved_run)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_history_db()
    settings.onnx_upload_dir.mkdir(parents=True, exist_ok=True)
    onnx_queue.configure(
        evaluate_fn=_run_onnx_job,
        timeout_seconds=settings.onnx_timeout_seconds,
    )
    onnx_queue.start()
    yield


app = FastAPI(title="Breast Density Benchmark API", version="0.3.0", lifespan=lifespan)

_cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"] + settings.allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history")
def get_history() -> dict[str, list[dict]]:
    return {"items": fetch_history()}


@app.get("/api/leaderboard")
def get_leaderboard() -> dict[str, list[dict]]:
    return {"items": fetch_leaderboard()}


# ── CSV evaluation (instant) ──

@app.post("/api/evaluate")
async def evaluate_model(
    model_name: str = Form(...),
    predictions: UploadFile = File(...),
) -> dict:
    _validate_upload_size(predictions, max_mb=settings.max_csv_upload_mb)

    filename = predictions.filename or ""
    if not filename.lower().endswith(".csv"):
        ext = Path(filename).suffix or "(no extension)"
        raise HTTPException(
            status_code=400,
            detail=f"Please upload a .csv file. Got: {ext}",
        )

    if not settings.ground_truth_csv.exists():
        raise HTTPException(
            status_code=500,
            detail="Ground-truth CSV not configured on the server.",
        )

    _check_model_name_unique(model_name.strip())

    try:
        result = evaluate_predictions(
            predictions_csv=predictions.file,
            ground_truth_csv=settings.ground_truth_csv,
            model_name=model_name,
            allowed_labels=settings.allowed_labels,
            include_quadratic_kappa=settings.enable_quadratic_kappa,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded file: {exc}",
        ) from exc

    saved_run = record_evaluation(
        result=result,
        submission_type="csv_predictions",
        source_filename=filename or "uploaded_predictions.csv",
    )

    return _evaluation_response(result=result, saved_run=saved_run)


# ── ONNX evaluation (queued) ──

@app.post("/api/submit-onnx")
async def submit_onnx_model(
    model_name: str = Form(...),
    model_file: UploadFile = File(...),
) -> dict:
    _validate_upload_size(model_file, max_mb=settings.max_onnx_upload_mb)

    filename = model_file.filename or ""
    if not filename.lower().endswith(".onnx"):
        raise HTTPException(
            status_code=400,
            detail="Model upload must be an .onnx file.",
        )

    if not settings.ground_truth_csv.exists():
        raise HTTPException(
            status_code=500,
            detail="Ground-truth CSV not configured on the server.",
        )

    _check_model_name_unique(model_name.strip())

    uploaded_bytes = await model_file.read()
    if len(uploaded_bytes) > _max_bytes(settings.max_onnx_upload_mb):
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded ONNX file is too large. Maximum allowed size is {settings.max_onnx_upload_mb} MB.",
        )

    # Pre-validate the ONNX file before saving to disk
    try:
        import onnxruntime as ort
        tmp_dir = tempfile.mkdtemp(prefix="onnx_validate_")
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(uploaded_bytes)
        try:
            sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
            input_meta = sess.get_inputs()[0]
            input_shape = list(input_meta.shape)
            expected = [1, settings.onnx_input_channels, settings.onnx_input_size, settings.onnx_input_size]
            for i, (actual, exp) in enumerate(zip(input_shape, expected)):
                if actual in (None, "batch", "N") and i == 0:
                    continue
                if isinstance(actual, str):
                    continue
                if actual != exp:
                    raise ValueError(
                        f"ONNX model input shape mismatch. "
                        f"Expected {expected}, got {input_shape}. "
                        f"The model must accept a grayscale tensor shaped "
                        f"[1, {settings.onnx_input_channels}, {settings.onnx_input_size}, {settings.onnx_input_size}]."
                    )
        except ValueError:
            raise
        except Exception as exc:
            error_str = str(exc)
            if "external data" in error_str.lower() or ".data" in error_str.lower():
                raise ValueError(
                    "This ONNX model requires an external .data file which "
                    "cannot be uploaded separately. Please re-export with "
                    "embedded weights. PyTorch: torch.onnx.export(model, ..., "
                    "'model.onnx') without external_data threshold. "
                    "ONNX: onnx.save(model, 'model.onnx', "
                    "save_as_external_data=False)."
                )
            raise ValueError(
                f"The uploaded .onnx file is corrupted or not a valid ONNX model. Details: {exc}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Save to disk (out of RAM)
    job_id = uuid.uuid4().hex[:12]
    dest_path = settings.onnx_upload_dir / f"{job_id}.onnx"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(uploaded_bytes)
    del uploaded_bytes

    submission = await onnx_queue.submit(
        model_name=model_name.strip(),
        model_path=dest_path,
        filename=filename,
    )

    return submission


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str) -> dict:
    status = onnx_queue.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return status


@app.get("/api/queue")
def get_queue() -> dict:
    return onnx_queue.get_queue_info()
