"""ONNX model inference with automatic GPU acceleration.

Provider priority (auto-detected at startup):
  1. PyTorch CUDA via onnx2torch   → converts ONNX graph to PyTorch, runs on GPU
  2. ONNX Runtime CUDAExecutionProvider  → if onnxruntime-gpu is installed
  3. ONNX Runtime CPUExecutionProvider   → fallback (multi-threaded)

Performance optimisations:
  - Preprocessed image tensors are cached to disk on first run (~3 GB .npy).
    Every subsequent model submission skips PNG load/resize entirely.
  - ONNX models with high opset (≥18) are automatically downgraded to opset 17
    so that onnx2torch can convert them and run on GPU.
  - CPU fallback uses all available cores via ORT threading settings.
"""

from __future__ import annotations

import logging
import os
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

_tensor_cache: dict[str, np.ndarray] | None = None


def clear_tensor_cache(cache_dir: Path | None = None) -> None:
    """Free the in-memory tensor cache and optionally delete the disk .npz file."""
    global _tensor_cache
    _tensor_cache = None
    logger.info("In-memory tensor cache cleared")
    if cache_dir and cache_dir.exists():
        for npz in cache_dir.glob("preprocessed_*.npz"):
            npz.unlink()
            logger.info("Deleted disk cache: %s", npz)


# ---------------------------------------------------------------------------
# Backend detection (runs once at import time)
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Return the best available inference backend name."""
    # 1. PyTorch with CUDA + onnx2torch
    try:
        import torch
        import onnx2torch  # noqa: F401

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("✓ GPU backend: PyTorch CUDA (%s) via onnx2torch", device_name)
            return "pytorch_cuda"
        else:
            logger.info("PyTorch found but no CUDA device available")
    except ImportError:
        logger.debug("onnx2torch or torch not available")

    # 2. ONNX Runtime with CUDA
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            logger.info("✓ GPU backend: ONNX Runtime CUDAExecutionProvider")
            return "ort_cuda"
    except ImportError:
        pass

    # 3. ONNX Runtime CPU
    try:
        import onnxruntime as ort  # noqa: F811

        if "CPUExecutionProvider" in ort.get_available_providers():
            logger.warning(
                "⚠ Using CPU-only backend — inference will be slower. "
                "Install PyTorch+CUDA or build onnxruntime-gpu for GPU acceleration."
            )
            return "ort_cpu"
    except ImportError:
        pass

    raise RuntimeError(
        "No inference backend available. Install either "
        "'torch' + 'onnx2torch' (with CUDA) or 'onnxruntime'."
    )


_BACKEND = _detect_backend()


# ---------------------------------------------------------------------------
# Tensor cache — preprocess all benchmark images ONCE
# ---------------------------------------------------------------------------

def _get_or_build_tensor_cache(
    *,
    manifest: pd.DataFrame,
    image_ids: list[str],
    image_root: Path,
    input_size: int,
    input_channels: int,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """Load cached tensors from disk, or build + save them on first call."""
    global _tensor_cache

    if _tensor_cache is not None and len(_tensor_cache) == len(image_ids):
        logger.info("Using in-memory tensor cache (%d images)", len(_tensor_cache))
        return _tensor_cache

    cache_file = cache_dir / f"preprocessed_{input_channels}ch_{input_size}px.npz"

    if cache_file.exists():
        logger.info("Loading preprocessed tensor cache from %s …", cache_file)
        t0 = time.perf_counter()
        data = np.load(str(cache_file))
        _tensor_cache = {k: data[k] for k in data.files}
        elapsed = time.perf_counter() - t0
        logger.info("Tensor cache loaded: %d images in %.1f s", len(_tensor_cache), elapsed)

        # Verify all image_ids are present
        missing = [iid for iid in image_ids if iid not in _tensor_cache]
        if not missing:
            return _tensor_cache
        logger.warning("%d images missing from cache, rebuilding…", len(missing))

    # Build cache from PNGs
    logger.info("Building tensor cache for %d images (one-time operation)…", len(image_ids))
    t0 = time.perf_counter()
    tensors: dict[str, np.ndarray] = {}

    for i, image_id in enumerate(image_ids):
        image_path = _resolve_image_path(image_id, manifest, image_root)
        tensor = _preprocess_image(
            image_path=image_path,
            input_size=input_size,
            input_channels=input_channels,
        )
        tensors[image_id] = tensor

        if (i + 1) % 200 == 0:
            logger.info("  Preprocessing: %d / %d images", i + 1, len(image_ids))

    # Save to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tensor cache to %s …", cache_file)
    np.savez(str(cache_file), **tensors)

    elapsed = time.perf_counter() - t0
    logger.info("Tensor cache built and saved in %.1f s", elapsed)

    _tensor_cache = tensors
    return _tensor_cache


# ---------------------------------------------------------------------------
# ONNX opset downgrade — makes onnx2torch compatible with more models
# ---------------------------------------------------------------------------

def _downgrade_opset_if_needed(model_path: Path, target_opset: int = 17) -> Path:
    """If the ONNX model uses opset > target_opset, convert it down.

    Returns the (possibly new) model path.
    """
    import onnx

    try:
        model = onnx.load(str(model_path))
    except Exception:
        # If external data is missing, try loading without it
        model = onnx.load(str(model_path), load_external_data=False)
    current_opset = model.opset_import[0].version if model.opset_import else 0

    if current_opset <= target_opset:
        logger.info("Model opset %d ≤ %d, no downgrade needed", current_opset, target_opset)
        return model_path

    logger.info("Downgrading model opset %d → %d for onnx2torch compatibility …",
                current_opset, target_opset)

    try:
        from onnx import version_converter
        converted = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(converted)

        # Save to a temp file next to the original
        downgraded_path = model_path.with_suffix(f".opset{target_opset}.onnx")
        onnx.save(converted, str(downgraded_path))
        logger.info("✓ Opset downgrade successful → %s", downgraded_path.name)
        return downgraded_path
    except Exception as exc:
        logger.warning("Opset downgrade failed (%s), using original model", exc)
        return model_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_onnx_benchmark(
    *,
    model_path: Path,
    image_manifest_csv: Path,
    image_root: Path,
    input_size: int,
    input_channels: int,
    allowed_labels: tuple[str, ...],
) -> StringIO:
    if input_channels not in (1, 3):
        raise ValueError("BENCHMARK_ONNX_INPUT_CHANNELS must be 1 or 3.")

    if not image_root.exists():
        raise ValueError(f"Private benchmark image folder not found at {image_root}.")

    manifest = _load_manifest(image_manifest_csv)
    image_ids = list(manifest["image_id"].dropna().astype(str).str.strip())
    expected_class_count = len(allowed_labels)

    # Build or load cached preprocessed tensors
    cache_dir = image_root.parent / "tensor_cache"
    cached_tensors = _get_or_build_tensor_cache(
        manifest=manifest,
        image_ids=image_ids,
        image_root=image_root,
        input_size=input_size,
        input_channels=input_channels,
        cache_dir=cache_dir,
    )

    logger.info(
        "Starting inference: %d images, backend=%s, input=%dx%dx%d",
        len(image_ids), _BACKEND, input_channels, input_size, input_size,
    )
    t0 = time.perf_counter()

    common_kwargs = dict(
        model_path=model_path,
        image_ids=image_ids,
        cached_tensors=cached_tensors,
        allowed_labels=allowed_labels,
        expected_class_count=expected_class_count,
    )

    if _BACKEND == "pytorch_cuda":
        # Try opset downgrade + onnx2torch GPU conversion
        try:
            downgraded_path = _downgrade_opset_if_needed(model_path)
            rows = _run_pytorch_cuda(**{**common_kwargs, "model_path": downgraded_path})
        except (NotImplementedError, RuntimeError, Exception) as exc:
            logger.warning(
                "⚠ onnx2torch conversion failed (%s: %s). "
                "Falling back to ONNX Runtime CPU (multi-threaded).",
                type(exc).__name__, exc,
            )
            rows = _run_ort(**common_kwargs, providers=["CPUExecutionProvider"])
        finally:
            # Clean up downgraded temp file
            downgraded = model_path.with_suffix(".opset17.onnx")
            if downgraded.exists() and downgraded != model_path:
                downgraded.unlink(missing_ok=True)
    elif _BACKEND == "ort_cuda":
        rows = _run_ort(
            **common_kwargs,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        rows = _run_ort(**common_kwargs, providers=["CPUExecutionProvider"])

    elapsed = time.perf_counter() - t0
    logger.info(
        "✓ Inference complete: %d images in %.1f s (%.2f img/s, backend=%s)",
        len(image_ids), elapsed, len(image_ids) / max(elapsed, 0.001), _BACKEND,
    )

    predictions = pd.DataFrame(rows)
    buffer = StringIO()
    predictions.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# PyTorch CUDA backend  (fastest path)
# ---------------------------------------------------------------------------

def _run_pytorch_cuda(
    *,
    model_path: Path,
    image_ids: list[str],
    cached_tensors: dict[str, np.ndarray],
    allowed_labels: tuple[str, ...],
    expected_class_count: int,
) -> list[dict[str, str]]:
    import torch
    from onnx2torch import convert

    device = torch.device("cuda")

    import onnx as _onnx
    try:
        onnx_model = _onnx.load(str(model_path))
        model_input = onnx_model.graph.input[0]
        input_shape = [
            d.dim_value if d.dim_value > 0 else None
            for d in model_input.type.tensor_type.shape.dim
        ]
        _validate_image_input_shape(
            input_shape=input_shape,
            input_channels=cached_tensors[image_ids[0]].shape[1],
            input_size=cached_tensors[image_ids[0]].shape[2],
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"The uploaded .onnx file is corrupted or not a valid ONNX model. Details: {exc}"
        ) from exc

    logger.info("Converting ONNX model to PyTorch …")
    model = convert(str(model_path))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded on GPU (%s)", torch.cuda.get_device_name(0))

    rows: list[dict[str, str]] = []
    batch_size = 16  # process multiple images at once for GPU throughput

    with torch.no_grad():
        for batch_start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[batch_start : batch_start + batch_size]

            # Load pre-cached tensors directly → skip all PNG I/O
            tensors = [torch.from_numpy(cached_tensors[iid]) for iid in batch_ids]
            batch_tensor = torch.cat(tensors, dim=0).to(device)

            # Forward pass on GPU
            output = model(batch_tensor)
            scores = output.cpu().numpy()

            # Map predictions
            predictions = _predicted_indexes_from_output(
                scores,
                expected_class_count=expected_class_count,
                batch_size=len(batch_ids),
            )
            for idx, predicted_index in enumerate(predictions):
                rows.append({
                    "image_id": batch_ids[idx],
                    "predicted_label": allowed_labels[predicted_index],
                })

            if batch_start % 100 < batch_size:
                logger.info(
                    "Progress: %d / %d images",
                    min(batch_start + batch_size, len(image_ids)),
                    len(image_ids),
                )

    return rows


# ---------------------------------------------------------------------------
# ONNX Runtime backend (CPU or CUDA) — multi-threaded
# ---------------------------------------------------------------------------

def _run_ort(
    *,
    model_path: Path,
    image_ids: list[str],
    cached_tensors: dict[str, np.ndarray],
    allowed_labels: tuple[str, ...],
    expected_class_count: int,
    providers: list[str],
) -> list[dict[str, str]]:
    import onnxruntime as ort

    num_cores = os.cpu_count() or 4

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use ALL CPU cores for parallel inference
    sess_options.intra_op_num_threads = num_cores
    sess_options.inter_op_num_threads = num_cores
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
    except Exception as exc:
        raise ValueError(
            "The uploaded .onnx file is corrupted or not a valid ONNX model. "
            f"Details: {exc}"
        ) from exc

    active_provider = session.get_providers()[0]
    logger.info(
        "ORT session: provider=%s, threads=%d",
        active_provider, num_cores,
    )

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    _validate_image_input_shape(
        input_shape=list(input_meta.shape),
        input_channels=cached_tensors[image_ids[0]].shape[1],
        input_size=cached_tensors[image_ids[0]].shape[2],
    )
    output_name = session.get_outputs()[0].name

    rows: list[dict[str, str]] = []
    for i, image_id in enumerate(image_ids):
        # Use pre-cached tensor — no PNG I/O
        tensor = cached_tensors[image_id]

        output = session.run([output_name], {input_name: tensor})[0]
        predicted_index = _predicted_indexes_from_output(
            output,
            expected_class_count=expected_class_count,
            batch_size=1,
        )[0]
        rows.append({"image_id": image_id, "predicted_label": allowed_labels[predicted_index]})

        if i % 100 == 0:
            logger.info("Progress: %d / %d images", i + 1, len(image_ids))

    return rows


def _predicted_indexes_from_output(
    output,
    *,
    expected_class_count: int,
    batch_size: int,
) -> list[int]:
    scores = np.asarray(output)

    if scores.ndim == 0:
        predictions = scores.reshape(1)
    elif scores.ndim == 1 and scores.shape[0] == batch_size:
        predictions = scores
    elif scores.ndim == 1 and batch_size == 1 and scores.shape[0] == expected_class_count:
        return [int(scores.argmax())]
    elif scores.ndim >= 2 and scores.shape[-1] == expected_class_count:
        flat_scores = scores.reshape(-1, expected_class_count)
        if flat_scores.shape[0] != batch_size:
            raise ValueError(
                f"ONNX model returned {flat_scores.shape[0]} score rows for {batch_size} image(s)."
            )
        return [int(index) for index in flat_scores.argmax(axis=1)]
    else:
        raise ValueError(
            "ONNX output must be either 4 class scores in A,B,C,D order "
            f"or integer class index 0=A,1=B,2=C,3=D. Got output shape {tuple(scores.shape)}."
        )

    predictions = predictions.reshape(-1)
    if predictions.shape[0] != batch_size:
        raise ValueError(
            f"ONNX model returned {predictions.shape[0]} label(s) for {batch_size} image(s)."
        )

    predicted_indexes = []
    for value in predictions:
        if np.issubdtype(predictions.dtype, np.floating) and not float(value).is_integer():
            raise ValueError(
                "ONNX label output must contain integer class index values: 0=A,1=B,2=C,3=D."
            )
        predicted_index = int(value)
        if predicted_index < 0 or predicted_index >= expected_class_count:
            raise ValueError(
                f"ONNX label output contains invalid class index {predicted_index}; expected 0-3."
            )
        predicted_indexes.append(predicted_index)

    return predicted_indexes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_manifest(image_manifest_csv: Path) -> pd.DataFrame:
    if not image_manifest_csv.exists():
        raise ValueError(f"Image manifest CSV not found at {image_manifest_csv}.")
    manifest = pd.read_csv(image_manifest_csv, dtype=str).rename(columns=str.strip)
    if "image_id" not in manifest.columns:
        raise ValueError("Image manifest CSV must contain an image_id column.")
    return manifest


def _validate_image_input_shape(
    *, input_shape: list, input_channels: int, input_size: int
) -> None:
    expected_shape = [1, input_channels, input_size, input_size]
    if len(input_shape) != len(expected_shape):
        raise ValueError(
            "ONNX model must accept image input shaped "
            f"{expected_shape}. Got input shape {input_shape}."
        )

    for index, (actual, expected) in enumerate(
        zip(input_shape, expected_shape, strict=True)
    ):
        if actual in (None, "batch", "N") and index == 0:
            continue
        if isinstance(actual, str):
            continue
        if actual != expected:
            raise ValueError(
                "ONNX model must accept image input shaped "
                f"{expected_shape}. Got input shape {input_shape}."
            )


def _resolve_image_path(
    image_id: str, manifest: pd.DataFrame, image_root: Path
) -> Path:
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
            candidates.append(
                raw_path if raw_path.is_absolute() else image_root / raw_path
            )
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

    raise ValueError(
        f"No private PNG/JPG benchmark image found for image_id {image_id}."
    )


def _preprocess_image(
    *,
    image_path: Path,
    input_size: int,
    input_channels: int,
) -> np.ndarray:
    mode = "L" if input_channels == 1 else "RGB"
    image = Image.open(image_path).convert(mode)
    image = image.resize((input_size, input_size))
    array = np.asarray(image, dtype=np.float32) / 255.0

    if input_channels == 1:
        array = array[np.newaxis, :, :]
    else:
        array = array.transpose(2, 0, 1)

    return array[np.newaxis, :, :, :]
