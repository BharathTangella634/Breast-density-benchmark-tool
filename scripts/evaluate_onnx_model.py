from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.evaluation import evaluate_predictions  # noqa: E402
from app.history import record_evaluation  # noqa: E402
from app.onnx_inference import run_onnx_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ONNX model against the private benchmark locally.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the .onnx model file.")
    parser.add_argument("--model-name", default=None, help="Name to show in the evaluation output.")
    parser.add_argument("--manifest", default=settings.image_manifest_csv, type=Path, help="Benchmark public manifest CSV.")
    parser.add_argument("--image-root", default=settings.image_root, type=Path, help="Private benchmark PNG/JPG folder.")
    parser.add_argument("--ground-truth", default=settings.ground_truth_csv, type=Path, help="Private ground-truth CSV.")
    parser.add_argument("--per-class", type=int, default=None, help="Optional quick balanced subset size per class.")
    parser.add_argument("--save-predictions", type=Path, default=None, help="Optional output CSV for generated predictions.")
    parser.add_argument("--save-history", action="store_true", help="Save this evaluation to the leaderboard history DB.")
    return parser.parse_args()


def build_subset_manifest(*, manifest_path: Path, ground_truth_path: Path, per_class: int) -> Path:
    manifest = pd.read_csv(manifest_path, dtype=str).rename(columns=str.strip)
    ground_truth = pd.read_csv(ground_truth_path, dtype=str).rename(columns=str.strip)
    merged = manifest.merge(ground_truth[["image_id", "true_label"]], on="image_id", how="inner")

    subset_frames = []
    for label in settings.allowed_labels:
        class_rows = merged.loc[merged["true_label"] == label].head(per_class)
        if len(class_rows) < per_class:
            raise SystemExit(f"Class {label} has only {len(class_rows)} rows; requested {per_class}.")
        subset_frames.append(class_rows)

    subset = pd.concat(subset_frames, ignore_index=True)
    subset = subset.drop(columns=["true_label"])
    temporary = tempfile.NamedTemporaryFile(prefix="onnx_subset_", suffix=".csv", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    subset.to_csv(temporary_path, index=False)
    return temporary_path


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model.stem

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    external_data = Path(f"{args.model}.data")
    if external_data.exists():
        print(f"Found external ONNX data file: {external_data}")

    manifest_path = args.manifest
    subset_manifest_path: Path | None = None
    if args.per_class is not None:
        subset_manifest_path = build_subset_manifest(
            manifest_path=args.manifest,
            ground_truth_path=args.ground_truth,
            per_class=args.per_class,
        )
        manifest_path = subset_manifest_path
        print(f"Using quick balanced subset: {args.per_class} per class ({args.per_class * len(settings.allowed_labels)} images)")

    start = perf_counter()
    predictions_csv = run_onnx_benchmark(
        model_path=args.model,
        image_manifest_csv=manifest_path,
        image_root=args.image_root,
        input_size=settings.onnx_input_size,
        input_channels=settings.onnx_input_channels,
        allowed_labels=settings.allowed_labels,
    )
    inference_seconds = perf_counter() - start

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(predictions_csv.getvalue(), encoding="utf-8")
        predictions_csv.seek(0)
        print(f"Saved predictions: {args.save_predictions}")

    result = evaluate_predictions(
        predictions_csv=predictions_csv,
        ground_truth_csv=args.ground_truth,
        model_name=model_name,
        allowed_labels=settings.allowed_labels,
        include_quadratic_kappa=settings.enable_quadratic_kappa,
    )
    total_seconds = perf_counter() - start

    print("\nEvaluation Results")
    print(f"Model: {result.model_name}")
    print(f"Samples: {result.sample_count}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Balanced accuracy: {result.balanced_accuracy:.4f}")
    print(f"Weighted F1: {result.weighted_f1:.4f}")
    print(f"Macro precision: {result.macro_precision:.4f}")
    print(f"Macro recall: {result.macro_recall:.4f}")
    print(f"Quadratic kappa: {'N/A' if result.quadratic_kappa is None else f'{result.quadratic_kappa:.4f}'}")
    print("Per-class F1:", {label: round(value, 4) for label, value in result.per_class_f1.items()})
    print(f"Inference time: {inference_seconds:.1f} seconds")
    print(f"Total time: {total_seconds:.1f} seconds ({total_seconds / 60:.2f} minutes)")

    if args.save_history:
        saved_run = record_evaluation(
            db_path=settings.history_db,
            result=result,
            submission_type="onnx_model_script",
            source_filename=args.model.name,
        )
        print(f"Saved to history as run #{saved_run['id']}")

    if subset_manifest_path:
        subset_manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
