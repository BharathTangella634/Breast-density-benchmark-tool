from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def run_onnx_benchmark(
    *,
    model_path: Path,
    image_manifest_csv: Path,
    image_root: Path,
    input_size: int,
    input_channels: int,
    allowed_labels: tuple[str, ...],
) -> StringIO:
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "ONNX evaluation dependencies are missing. Install backend requirements: onnxruntime, Pillow, and numpy."
        ) from exc

    if input_channels not in (1, 3):
        raise ValueError("BENCHMARK_ONNX_INPUT_CHANNELS must be 1 or 3.")

    if not image_manifest_csv.exists():
        raise ValueError(f"Image manifest CSV not found at {image_manifest_csv}.")

    if not image_root.exists():
        raise ValueError(f"Private benchmark image folder not found at {image_root}.")

    manifest = pd.read_csv(image_manifest_csv, dtype=str).rename(columns=str.strip)
    if "image_id" not in manifest.columns:
        raise ValueError("Image manifest CSV must contain an image_id column.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    expected_class_count = len(allowed_labels)

    rows: list[dict[str, str]] = []
    for image_id in manifest["image_id"].dropna():
        image_path = _resolve_image_path(str(image_id).strip(), manifest, image_root)
        tensor = _preprocess_image(
            image_path=image_path,
            image_module=Image,
            numpy_module=np,
            input_size=input_size,
            input_channels=input_channels,
        )
        output = session.run([output_name], {input_name: tensor})[0]
        scores = np.asarray(output)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if scores.shape[-1] != expected_class_count:
            raise ValueError(
                f"ONNX model output must have {expected_class_count} class scores in A,B,C,D order. "
                f"Got output shape {tuple(scores.shape)}."
            )
        predicted_index = int(scores[0].argmax())
        rows.append({"image_id": image_id, "prediction": allowed_labels[predicted_index]})

    predictions = pd.DataFrame(rows)
    buffer = StringIO()
    predictions.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def _resolve_image_path(image_id: str, manifest: pd.DataFrame, image_root: Path) -> Path:
    row = manifest.loc[manifest["image_id"] == image_id].head(1)
    candidate_values: list[str] = []
    for column in ("png_path", "image_path", "relative_path"):
        if column in row.columns and not row.empty:
            value = row.iloc[0].get(column)
            if isinstance(value, str) and value.strip():
                candidate_values.append(value.strip())

    candidates: list[Path] = []
    for value in candidate_values:
        raw_path = Path(value)
        if raw_path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(raw_path if raw_path.is_absolute() else image_root / raw_path)
        else:
            for extension in IMAGE_EXTENSIONS:
                candidates.append((image_root / raw_path).with_suffix(extension))

    for extension in IMAGE_EXTENSIONS:
        candidates.append(image_root / f"{image_id}{extension}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches = []
    for extension in IMAGE_EXTENSIONS:
        recursive_matches.extend(image_root.rglob(f"{image_id}{extension}"))
    if recursive_matches:
        return recursive_matches[0]

    raise ValueError(f"No private PNG/JPG benchmark image found for image_id {image_id}.")


def _preprocess_image(
    *,
    image_path: Path,
    image_module,
    numpy_module,
    input_size: int,
    input_channels: int,
):
    mode = "L" if input_channels == 1 else "RGB"
    image = image_module.open(image_path).convert(mode)
    image = image.resize((input_size, input_size))
    array = numpy_module.asarray(image, dtype=numpy_module.float32) / 255.0

    if input_channels == 1:
        array = array[numpy_module.newaxis, :, :]
    else:
        array = array.transpose(2, 0, 1)

    return array[numpy_module.newaxis, :, :, :]
