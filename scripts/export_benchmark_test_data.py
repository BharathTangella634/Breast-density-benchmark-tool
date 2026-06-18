#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

try:
    import pydicom
except ImportError:  # pragma: no cover - runtime environment check
    pydicom = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and verify benchmark test DICOM files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmark_prep/benchmark_test_public.csv"),
        help="Public benchmark CSV listing the files to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tanuh/benchmark_test_data"),
        help="Destination folder to create.",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Rename an existing output folder before creating the clean export.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dicom(path: Path) -> str:
    if pydicom is None:
        return "skipped_no_pydicom"

    dataset = pydicom.dcmread(path, stop_before_pixels=False)
    if not getattr(dataset, "SOPClassUID", None):
        raise ValueError("Missing SOPClassUID")
    if "PixelData" not in dataset:
        raise ValueError("Missing PixelData")
    return "ok"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"image_id", "dataset", "image_path", "relative_path"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing))}")
    return rows


def backup_existing_output(path: Path) -> None:
    if not path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}_backup_{timestamp}")
    path.rename(backup_path)
    print(f"Existing output moved to {backup_path}")


def write_portable_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_id", "case_id", "dataset", "image_path", "relative_path", "view_position", "laterality"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            portable = {key: row.get(key, "") for key in fieldnames}
            portable["image_path"] = str(Path(row["dataset"]) / row["relative_path"])
            writer.writerow(portable)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)

    if args.backup_existing:
        backup_existing_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verification_rows: list[dict[str, str]] = []
    failures: list[str] = []

    for index, row in enumerate(rows, start=1):
        src = Path(row["image_path"])
        dst = args.output_dir / row["dataset"] / row["relative_path"]

        if not src.exists():
            failures.append(f"{row['image_id']}: missing source {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        src_hash = sha256(src)
        dst_hash = sha256(dst)
        if src_hash != dst_hash:
            failures.append(f"{row['image_id']}: checksum mismatch")
            dicom_status = "not_checked"
        else:
            try:
                dicom_status = validate_dicom(dst)
            except Exception as exc:  # pragma: no cover - validation report
                dicom_status = f"failed: {exc}"
                failures.append(f"{row['image_id']}: DICOM validation failed: {exc}")

        verification_rows.append(
            {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "source_path": str(src),
                "export_path": str(dst.relative_to(args.output_dir)),
                "sha256": dst_hash,
                "dicom_status": dicom_status,
            }
        )

        if index % 100 == 0:
            print(f"Verified {index}/{len(rows)} files")

    write_portable_manifest(args.output_dir / "benchmark_test_public.csv", rows)

    report_path = args.output_dir / "verification_report.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "dataset", "source_path", "export_path", "sha256", "dicom_status"],
        )
        writer.writeheader()
        writer.writerows(verification_rows)

    with (args.output_dir / "README.txt").open("w") as handle:
        handle.write("Breast density benchmark test data\n")
        handle.write("Images are organized under EMBED/ and IBIA/.\n")
        handle.write("Use benchmark_test_public.csv for image_id values and portable image paths.\n")
        handle.write("Intern prediction labels should be A, B, C, or D.\n")
        handle.write("verification_report.csv contains checksum and DICOM-read validation results.\n")

    if failures:
        print("\n".join(failures[:20]))
        raise SystemExit(f"Export completed with {len(failures)} failure(s).")

    print(f"Exported and verified {len(verification_rows)} files in {args.output_dir}")
    print(f"Verification report: {report_path}")


if __name__ == "__main__":
    main()
