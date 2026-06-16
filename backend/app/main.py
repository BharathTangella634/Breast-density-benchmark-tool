from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.evaluation import evaluate_predictions
from app.history import fetch_history, fetch_leaderboard, initialize_history_db, record_evaluation


app = FastAPI(title="Breast Density Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_history_db(settings.history_db)


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
