from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid


# Example:
# /home/tanuh/miniconda3/envs/pytorch_env/bin/python scripts/convert_any_format_to_dcm.py \
#   --input /home/tanuh/Downloads/benchmark_200 \
#   --output-dir /home/tanuh/Downloads/benchmark_200_dcm_clean \
#   --extensions .bin .dicom .dcm \
#   --validate-pixels

DEFAULT_EXTENSIONS = (".bin", ".dicom", ".dcm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DICOM-content files with nonstandard extensions, such as .bin or .dicom, "
            "to standards-shaped .dcm files. This reads and rewrites the DICOM dataset; it is "
            "not a plain file rename."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input file or folder.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for converted .dcm files.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Extensions to include when --input is a folder. Default: .bin .dicom .dcm",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input folders recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .dcm files.")
    parser.add_argument(
        "--validate-pixels",
        action="store_true",
        help="Also decode pixel data before writing, to catch unreadable mammogram pixels.",
    )
    parser.add_argument(
        "--report-name",
        default="dcm_conversion_report.csv",
        help="CSV report filename written inside output-dir.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: list[str]) -> set[str]:
    normalized = set()
    for extension in extensions:
        extension = extension.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)
    return normalized


def iter_inputs(input_path: Path, extensions: set[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def ensure_file_meta(dataset: pydicom.Dataset) -> None:
    if not hasattr(dataset, "file_meta") or dataset.file_meta is None:
        dataset.file_meta = FileMetaDataset()

    file_meta = dataset.file_meta
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if not transfer_syntax:
        transfer_syntax = ImplicitVRLittleEndian
        file_meta.TransferSyntaxUID = transfer_syntax

    sop_class_uid = getattr(dataset, "SOPClassUID", None)
    sop_instance_uid = getattr(dataset, "SOPInstanceUID", None)

    if sop_class_uid and not getattr(file_meta, "MediaStorageSOPClassUID", None):
        file_meta.MediaStorageSOPClassUID = sop_class_uid
    if sop_instance_uid and not getattr(file_meta, "MediaStorageSOPInstanceUID", None):
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    if not getattr(file_meta, "ImplementationClassUID", None):
        file_meta.ImplementationClassUID = generate_uid()

    if transfer_syntax == ExplicitVRLittleEndian:
        dataset.is_implicit_VR = False
        dataset.is_little_endian = True
    else:
        dataset.is_implicit_VR = True
        dataset.is_little_endian = True

    if not getattr(dataset, "SOPClassUID", None) and getattr(file_meta, "MediaStorageSOPClassUID", None):
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    if not getattr(dataset, "SOPInstanceUID", None) and getattr(file_meta, "MediaStorageSOPInstanceUID", None):
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID


def convert_to_dcm(source_path: Path, output_path: Path, validate_pixels: bool, overwrite: bool) -> tuple[str, str]:
    if output_path.exists() and not overwrite:
        return "skipped", "output exists; pass --overwrite to replace it"

    dataset = pydicom.dcmread(source_path, force=True)
    ensure_file_meta(dataset)

    if validate_pixels:
        _ = dataset.pixel_array

    dataset.save_as(output_path, write_like_original=False)

    # Confirm the rewritten file can be read as a normal DICOM file, without force=True.
    pydicom.dcmread(output_path, force=False, stop_before_pixels=True)
    return "ok", ""


def main() -> None:
    args = parse_args()
    extensions = normalize_extensions(args.extensions)
    input_paths = iter_inputs(args.input, extensions, args.recursive)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, str]] = []

    for source_path in input_paths:
        output_path = args.output_dir / f"{source_path.stem}.dcm"
        try:
            status, message = convert_to_dcm(
                source_path=source_path,
                output_path=output_path,
                validate_pixels=args.validate_pixels,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - batch tool reports every failed file.
            status = "error"
            message = str(exc)

        report_rows.append(
            {
                "source_path": str(source_path),
                "source_extension": source_path.suffix.lower(),
                "output_path": str(output_path),
                "status": status,
                "message": message,
            }
        )

    report_path = args.output_dir / args.report_name
    with report_path.open("w", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=["source_path", "source_extension", "output_path", "status", "message"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    ok_count = sum(row["status"] == "ok" for row in report_rows)
    skipped_count = sum(row["status"] == "skipped" for row in report_rows)
    error_count = sum(row["status"] == "error" for row in report_rows)

    print(f"Input files: {len(report_rows)}")
    print(f"Converted: {ok_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Report: {report_path}")

    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
