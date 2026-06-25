import logging
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Configure logging before importing inference (which detects backend at import time)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from app.config import settings
from app.evaluation import evaluate_predictions
from app.history import fetch_history, fetch_leaderboard, initialize_history_db, record_evaluation
from app.onnx_inference import run_onnx_benchmark


app = FastAPI(title="Breast Density Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_history_db(settings.history_db)


def _max_bytes(megabytes: int) -> int:
    return megabytes * 1024 * 1024


def _validate_upload_size(upload: UploadFile, *, max_mb: int) -> None:
    size = getattr(upload, "size", None)
    if size is not None and size > _max_bytes(max_mb):
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file is too large. Maximum allowed size is {max_mb} MB.",
        )


def _evaluation_response(*, result, saved_run: dict) -> dict:
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


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history")
def get_history() -> dict[str, list[dict]]:
    return {"items": fetch_history(settings.history_db)}


@app.get("/api/leaderboard")
def get_leaderboard() -> dict[str, list[dict]]:
    return {"items": fetch_leaderboard(settings.history_db)}


@app.post("/api/evaluate")
async def evaluate_model(
    model_name: str = Form(...),
    predictions: UploadFile = File(...),
) -> dict:
    _validate_upload_size(predictions, max_mb=settings.max_csv_upload_mb)

    if not settings.ground_truth_csv.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Ground-truth CSV not found at {settings.ground_truth_csv}. Set BENCHMARK_GROUND_TRUTH_CSV.",
        )

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

    saved_run = record_evaluation(
        db_path=settings.history_db,
        result=result,
        submission_type="csv_predictions",
        source_filename=predictions.filename or "uploaded_predictions.csv",
    )

    return _evaluation_response(result=result, saved_run=saved_run)


@app.post("/api/evaluate-onnx")
async def evaluate_onnx_model(
    model_name: str = Form(...),
    model_file: UploadFile = File(...),
) -> dict:
    _validate_upload_size(model_file, max_mb=settings.max_onnx_upload_mb)

    filename = model_file.filename or ""
    if not filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Model upload must be an .onnx file.")

    if not settings.ground_truth_csv.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Ground-truth CSV not found at {settings.ground_truth_csv}. Set BENCHMARK_GROUND_TRUTH_CSV.",
        )

    import tempfile
    import shutil

    # Save uploaded file to a dedicated temp directory (not just a single file)
    # so we can handle ONNX models with external data (.onnx.data companion files).
    tmp_dir = tempfile.mkdtemp(prefix="onnx_eval_")
    tmp_model_path = Path(tmp_dir) / filename

    try:
        uploaded_bytes = await model_file.read()
        if len(uploaded_bytes) > _max_bytes(settings.max_onnx_upload_mb):
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded ONNX file is too large. Maximum allowed size is {settings.max_onnx_upload_mb} MB.",
            )
        tmp_model_path.write_bytes(uploaded_bytes)

        # If the model uses external data (weights in a separate file),
        # convert it to a self-contained model with all weights embedded.
        try:
            import onnx
            model_proto = onnx.load(str(tmp_model_path), load_external_data=False)

            # Check if any tensor references external data
            has_external = any(
                tensor.HasField("data_location")
                and tensor.data_location == onnx.TensorProto.EXTERNAL
                for tensor in model_proto.graph.initializer
            )

            if has_external:
                # Try to load with external data (will fail if .data file missing)
                # In that case, tell the user to embed weights before uploading.
                try:
                    model_proto = onnx.load(str(tmp_model_path))
                except Exception:
                    raise ValueError(
                        "This ONNX model stores weights in an external .data file "
                        "which was not uploaded. Please re-export your model with "
                        "weights embedded. In PyTorch: "
                        "torch.onnx.export(...) without size thresholds, or use "
                        "onnx.save(model, 'out.onnx', save_as_external_data=False)."
                    )

                # Re-save as self-contained (all weights inside the .onnx file)
                selfcontained_path = Path(tmp_dir) / "model_selfcontained.onnx"
                onnx.save(
                    model_proto,
                    str(selfcontained_path),
                    save_as_external_data=False,
                )
                tmp_model_path = selfcontained_path
        except ValueError:
            raise
        except Exception as load_exc:
            logging.getLogger(__name__).debug("ONNX pre-check skipped: %s", load_exc)

        predictions_csv = await asyncio.wait_for(
            asyncio.to_thread(
                run_onnx_benchmark,
                model_path=tmp_model_path,
                image_manifest_csv=settings.image_manifest_csv,
                image_root=settings.image_root,
                input_size=settings.onnx_input_size,
                input_channels=settings.onnx_input_channels,
                allowed_labels=settings.allowed_labels,
            ),
            timeout=settings.onnx_timeout_seconds,
        )
        result = evaluate_predictions(
            predictions_csv=predictions_csv,
            ground_truth_csv=settings.ground_truth_csv,
            model_name=model_name,
            allowed_labels=settings.allowed_labels,
            include_quadratic_kappa=settings.enable_quadratic_kappa,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "ONNX evaluation timed out before completion. "
                "Please submit a prediction CSV, or try a smaller/faster standalone ONNX pipeline."
            ),
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    saved_run = record_evaluation(
        db_path=settings.history_db,
        result=result,
        submission_type="onnx_model",
        source_filename=filename,
    )

    return _evaluation_response(result=result, saved_run=saved_run)
