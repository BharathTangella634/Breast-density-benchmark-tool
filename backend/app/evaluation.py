from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)


REQUIRED_PREDICTION_COLUMNS = {"image_id", "predicted_label"}
REQUIRED_GROUND_TRUTH_COLUMNS = {"image_id", "true_label"}
SUBJECT_ID_PATTERN = re.compile(r"^subject_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    sample_count: int
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    weighted_f1: float
    quadratic_kappa: float | None
    per_class_f1: dict[str, float]


def _read_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, dtype=str).rename(columns=str.strip)


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required column(s): {columns}")


def _subject_id(value: str) -> str | None:
    match = SUBJECT_ID_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return f"subject_{int(match.group(1)):04d}"


def _require_subject_ids(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    frame = frame.copy()
    normalized_ids = frame["image_id"].map(lambda value: _subject_id(str(value)))
    invalid_ids = frame.loc[normalized_ids.isna(), "image_id"]
    if not invalid_ids.empty:
        examples = ", ".join(invalid_ids.astype(str).head(5))
        raise ValueError(
            f"{name} image_id values must use subject_#### format "
            f"(e.g. subject_0001, subject_0002). Invalid example(s): {examples}"
        )
    frame["image_id"] = normalized_ids
    return frame


def _validate_predictions(frame: pd.DataFrame, *, allowed_labels: tuple[str, ...]) -> pd.DataFrame:
    """Validate and return predictions with only image_id and predicted_label columns."""
    frame = frame[["image_id", "predicted_label"]].copy()
    frame["image_id"] = frame["image_id"].str.strip()
    frame["predicted_label"] = frame["predicted_label"].str.strip().str.upper()
    frame = _require_subject_ids(frame, name="Predictions CSV")

    empty_rows = frame["predicted_label"].isna() | (frame["predicted_label"] == "")
    if empty_rows.any():
        count = int(empty_rows.sum())
        raise ValueError(f"CSV has {count} row(s) with empty predicted_label values.")

    duplicates = frame[frame["image_id"].duplicated(keep=False)]
    if not duplicates.empty:
        examples = ", ".join(duplicates["image_id"].unique()[:5])
        raise ValueError(f"CSV has duplicate image_id values: {examples}")

    unknown = set(frame["predicted_label"]).difference(allowed_labels)
    if unknown:
        labels = ", ".join(sorted(unknown))
        raise ValueError(
            f"predicted_label must be one of {', '.join(allowed_labels)}. "
            f"Found unsupported label(s): {labels}"
        )

    return frame


def evaluate_predictions(
    *,
    predictions_csv,
    ground_truth_csv: Path,
    model_name: str,
    allowed_labels: tuple[str, ...],
    include_quadratic_kappa: bool = True,
) -> EvaluationResult:
    """Score an uploaded predictions CSV against the private local labels."""

    try:
        predictions = _read_csv(predictions_csv)
    except Exception:
        raise ValueError(
            "Could not parse the uploaded file as CSV. "
            "Please upload a valid CSV with columns: image_id, predicted_label"
        )

    if predictions.empty:
        raise ValueError("Uploaded CSV is empty.")

    ground_truth = _read_csv(ground_truth_csv)

    _validate_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "Predictions CSV")
    _validate_columns(ground_truth, REQUIRED_GROUND_TRUTH_COLUMNS, "Ground-truth CSV")

    predictions = _validate_predictions(predictions, allowed_labels=allowed_labels)
    ground_truth = ground_truth[["image_id", "true_label"]].dropna()
    ground_truth = _require_subject_ids(ground_truth, name="Ground-truth CSV")

    expected_count = len(ground_truth)
    merged = ground_truth.merge(predictions, on="image_id", how="inner")

    if merged.empty:
        raise ValueError(
            "No image_id values matched the benchmark set. "
            "Make sure your CSV uses the exact image_id values from the public manifest "
            "(e.g. subject_0001, subject_0002)."
        )

    if len(merged) < expected_count:
        missing_ids = set(ground_truth["image_id"]) - set(predictions["image_id"])
        examples = ", ".join(sorted(missing_ids)[:5])
        raise ValueError(
            f"CSV has predictions for only {len(merged)} of {expected_count} benchmark images. "
            f"All {expected_count} image_id values are required. "
            f"Missing examples: {examples}"
        )

    y_true = merged["true_label"]
    y_pred = merged["predicted_label"]

    per_class_values = f1_score(
        y_true,
        y_pred,
        labels=list(allowed_labels),
        average=None,
        zero_division=0,
    )

    quadratic_kappa = None
    if include_quadratic_kappa:
        quadratic_kappa = cohen_kappa_score(y_true, y_pred, labels=list(allowed_labels), weights="quadratic")

    return EvaluationResult(
        model_name=model_name,
        sample_count=int(len(merged)),
        macro_f1=float(f1_score(y_true, y_pred, labels=list(allowed_labels), average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=list(allowed_labels), average="weighted", zero_division=0)),
        quadratic_kappa=None if quadratic_kappa is None else float(quadratic_kappa),
        per_class_f1={label: float(score) for label, score in zip(allowed_labels, per_class_values)},
    )
