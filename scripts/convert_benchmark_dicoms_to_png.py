from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.uid import ImplicitVRLittleEndian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert private benchmark DICOMs to 1024x1024 PNGs.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with image_id and image_path columns.")
    parser.add_argument("--dicom-root", required=True, type=Path, help="Folder containing the benchmark DICOM tree.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private output folder for PNG files.")
    parser.add_argument("--size", default=1024, type=int, help="Output PNG width and height.")
    return parser.parse_args()


def dicom_to_uint8(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    if not getattr(dataset.file_meta, "TransferSyntaxUID", None):
        dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    pixels = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    lower, upper = np.percentile(pixels, (1, 99))
    if upper <= lower:
        lower, upper = float(pixels.min()), float(pixels.max())
    if upper <= lower:
        return np.zeros_like(pixels, dtype=np.uint8)

    pixels = np.clip((pixels - lower) / (upper - lower), 0, 1)
    return (pixels * 255).astype(np.uint8)


def resolve_dicom_path(row: pd.Series, dicom_root: Path) -> Path:
    image_path = Path(str(row["image_path"]))
    if image_path.is_absolute():
        return image_path
    return dicom_root / image_path


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str).rename(columns=str.strip)
    required_columns = {"image_id", "image_path"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise SystemExit(f"Manifest missing required columns: {sorted(missing_columns)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, str]] = []

    for row in manifest.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        image_id = str(row_series["image_id"]).strip()
        dicom_path = resolve_dicom_path(row_series, args.dicom_root)
        png_path = args.output_dir / f"{image_id}.png"

        try:
            pixels = dicom_to_uint8(dicom_path)
            image = Image.fromarray(pixels, mode="L").resize((args.size, args.size), Image.Resampling.LANCZOS)
            image.save(png_path)
            status = "ok"
            message = ""
        except Exception as exc:  # noqa: BLE001 - report all conversion failures without stopping the batch.
            status = "error"
            message = str(exc)

        report_rows.append(
            {
                "image_id": image_id,
                "dicom_path": str(dicom_path),
                "png_path": str(png_path),
                "status": status,
                "message": message,
            }
        )

    report = pd.DataFrame(report_rows)
    report_path = args.output_dir / "conversion_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Converted: {(report['status'] == 'ok').sum()} / {len(report)}")
    print(f"Report: {report_path}")

    failed = report[report["status"] != "ok"]
    if not failed.empty:
        print(f"Failures: {len(failed)}")
        print(failed.head(10).to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
