from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


REQUIRED_PREDICTION_COLUMNS = {"image_id"}
REQUIRED_GROUND_TRUTH_COLUMNS = {"image_id", "true_label"}


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    sample_count: int
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    quadratic_kappa: float | None
    per_class_f1: dict[str, float]


def _read_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, dtype=str).rename(columns=str.strip)


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required column(s): {columns}")


def _normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if "predicted_label" in frame.columns:
        return frame[["image_id", "predicted_label"]]

    if "prediction" in frame.columns:
        normalized = frame[["image_id", "prediction"]].rename(columns={"prediction": "predicted_label"})
        return normalized

    probability_columns = ["p0", "p1", "p2", "p3"]
    if all(column in frame.columns for column in probability_columns):
        probability_values = frame[probability_columns].apply(pd.to_numeric, errors="coerce")
        if probability_values.isna().any().any():
            raise ValueError("Probability submission contains non-numeric values in p0,p1,p2,p3.")

        label_names = ["A", "B", "C", "D"]
        predicted_indices = probability_values.to_numpy().argmax(axis=1)
        normalized = frame[["image_id"]].copy()
        normalized["predicted_label"] = [label_names[index] for index in predicted_indices]
        return normalized

    raise ValueError("Predictions CSV must contain predicted_label, prediction, or p0,p1,p2,p3.")


def evaluate_predictions(
    *,
    predictions_csv,
    ground_truth_csv: Path,
    model_name: str,
    allowed_labels: tuple[str, ...],
    include_quadratic_kappa: bool = True,
) -> EvaluationResult:
    """Score an uploaded predictions CSV against the private local labels."""

    predictions = _read_csv(predictions_csv)
    ground_truth = _read_csv(ground_truth_csv)

    _validate_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "Predictions CSV")
    _validate_columns(ground_truth, REQUIRED_GROUND_TRUTH_COLUMNS, "Ground-truth CSV")

    predictions = _normalize_predictions(predictions).dropna()
    ground_truth = ground_truth[["image_id", "true_label"]].dropna()

    merged = ground_truth.merge(predictions, on="image_id", how="inner")
    if merged.empty:
        raise ValueError("No prediction rows matched the private ground-truth image_id values.")

    unknown = set(merged["predicted_label"]).difference(allowed_labels)
    if unknown:
        labels = ", ".join(sorted(unknown))
        raise ValueError(f"Predictions contain unsupported label(s): {labels}")

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
        macro_precision=float(
            precision_score(y_true, y_pred, labels=list(allowed_labels), average="macro", zero_division=0)
        ),
        macro_recall=float(recall_score(y_true, y_pred, labels=list(allowed_labels), average="macro", zero_division=0)),
        quadratic_kappa=None if quadratic_kappa is None else float(quadratic_kappa),
        per_class_f1={label: float(score) for label, score in zip(allowed_labels, per_class_values)},
    )
