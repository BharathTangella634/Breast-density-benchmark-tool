#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


EMBED_LABEL_MAP = {
    "1.0": "A",
    "2.0": "B",
    "3.0": "C",
    "4.0": "D",
}

IBIA_LABEL_MAP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EMBED and IBIA benchmark CSV files.")
    parser.add_argument(
        "--embed-metadata",
        type=Path,
        default=Path("/home/tanuh/EMBED/tables/EMBED_OpenData_metadata.csv"),
    )
    parser.add_argument(
        "--embed-clinical",
        type=Path,
        default=Path("/home/tanuh/EMBED/tables/EMBED_OpenData_clinical_reduced.csv"),
    )
    parser.add_argument(
        "--ibia-root",
        type=Path,
        default=Path("/home/tanuh/a_data"),
    )
    parser.add_argument(
        "--ibia-metadata",
        type=Path,
        default=Path("/home/tanuh/a_data/mamos.csv"),
    )
    parser.add_argument(
        "--embed-root",
        type=Path,
        default=Path("/home/tanuh/EMBED"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark_prep"),
    )
    parser.add_argument("--embed-count", type=int, default=300)
    parser.add_argument("--ibia-count", type=int, default=500)
    parser.add_argument("--balanced-per-class", type=int, default=200)
    parser.add_argument("--target-ibia-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_blank(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "none"}


def read_embed_labels(clinical_csv: Path) -> dict[tuple[str, str, str], str]:
    labels_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    with clinical_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            density = (row.get("tissueden") or "").strip()
            if density not in EMBED_LABEL_MAP:
                continue
            key = (
                (row.get("empi_anon") or "").strip(),
                (row.get("acc_anon") or "").strip(),
                (row.get("cohort_num") or "").strip(),
            )
            if all(key):
                labels_by_key[key].add(density)

    output: dict[tuple[str, str, str], str] = {}
    for key, values in labels_by_key.items():
        if len(values) == 1:
            output[key] = EMBED_LABEL_MAP[next(iter(values))]
    return output


def filtered_embed_rows(
    metadata_csv: Path,
    labels_by_key: dict[tuple[str, str, str], str],
    embed_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    with metadata_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                (row.get("empi_anon") or "").strip(),
                (row.get("acc_anon") or "").strip(),
                (row.get("cohort_num") or "").strip(),
            )
            if key not in labels_by_key:
                continue

            if (row.get("FinalImageType") or "").strip().upper() != "2D":
                continue
            if (row.get("ViewPosition") or "").strip().upper() not in {"CC", "MLO"}:
                continue
            if not is_blank(row.get("spot_mag")):
                continue
            if (row.get("PatientSex") or "").strip().upper() != "F":
                continue
            if (row.get("BreastImplantPresent") or "").strip().upper() != "NO":
                continue

            try:
                xray_tube_current = float((row.get("XRayTubeCurrent") or "").strip())
                bits_stored = float((row.get("BitsStored") or "").strip())
            except ValueError:
                continue

            if xray_tube_current < 100:
                continue
            if bits_stored != 12.0:
                continue

            anon_dicom_path = (row.get("anon_dicom_path") or "").strip()
            if not anon_dicom_path or anon_dicom_path in seen_paths:
                continue

            seen_paths.add(anon_dicom_path)
            rows.append(
                {
                    "image_id": "",
                    "case_id": key[0],
                    "dataset": "EMBED",
                    "image_path": str(embed_root / anon_dicom_path),
                    "relative_path": anon_dicom_path,
                    "view_position": (row.get("ViewPosition") or "").strip(),
                    "laterality": (row.get("ImageLateralityFinal") or "").strip(),
                    "true_label": labels_by_key[key],
                }
            )

    return rows


def read_ibia_labels(ibia_metadata_csv: Path) -> dict[str, str]:
    labels_by_relative_path: dict[str, str] = {}

    with ibia_metadata_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = (row.get("Image File Name (with path)") or "").strip()
            density = (row.get("Breast density category") or "").strip().upper()
            if relative_path and density in IBIA_LABEL_MAP:
                cleaned_path = relative_path.removeprefix("a_data/")
                labels_by_relative_path[cleaned_path] = IBIA_LABEL_MAP[density]

                filename = Path(cleaned_path).name
                case_prefix = filename.split("_", 1)[0]
                normalized_path = f"{case_prefix}/{filename}"
                labels_by_relative_path[normalized_path] = IBIA_LABEL_MAP[density]

    return labels_by_relative_path


def ibia_rows(ibia_root: Path, labels_by_relative_path: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(ibia_root.rglob("*.dcm")):
        case_id = path.parent.name
        relative_path = str(path.relative_to(ibia_root))
        rows.append(
            {
                "image_id": "",
                "case_id": case_id,
                "dataset": "IBIA",
                "image_path": str(path),
                "relative_path": relative_path,
                "view_position": "",
                "laterality": "",
                "true_label": labels_by_relative_path.get(relative_path, ""),
            }
        )
    return rows


def sample_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows but only found {len(rows)}.")
    rng = random.Random(seed)
    sampled = rows[:]
    rng.shuffle(sampled)
    return sampled[:count]


def balanced_sample_rows(
    *,
    embed_rows: list[dict],
    ibia_rows_: list[dict],
    per_class: int,
    target_ibia_per_class: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    selected_embed: list[dict] = []
    selected_ibia: list[dict] = []

    for label in ["A", "B", "C", "D"]:
        embed_pool = [row for row in embed_rows if row["true_label"] == label]
        ibia_pool = [row for row in ibia_rows_ if row["true_label"] == label]
        rng.shuffle(embed_pool)
        rng.shuffle(ibia_pool)

        ibia_take = min(target_ibia_per_class, len(ibia_pool), per_class)
        embed_take = per_class - ibia_take
        if embed_take > len(embed_pool):
            raise ValueError(
                f"Not enough EMBED rows to fill class {label}: "
                f"need {embed_take}, found {len(embed_pool)}."
            )

        selected_ibia.extend(ibia_pool[:ibia_take])
        selected_embed.extend(embed_pool[:embed_take])

    rng.shuffle(selected_embed)
    rng.shuffle(selected_ibia)
    return selected_embed, selected_ibia


def assign_image_ids(rows: list[dict], prefix: str) -> list[dict]:
    output = []
    for index, row in enumerate(rows, start=1):
        enriched = dict(row)
        enriched["image_id"] = f"{prefix}_{index:04d}"
        output.append(enriched)
    return output


def write_csv(path: Path, rows: list[dict], include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "case_id",
        "dataset",
        "image_path",
        "relative_path",
        "view_position",
        "laterality",
    ]
    if include_labels:
        fieldnames.append("true_label")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(payload)


def write_combined_labels(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "true_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image_id": row["image_id"], "true_label": row["true_label"]})


def main() -> None:
    args = parse_args()

    embed_labels = read_embed_labels(args.embed_clinical)
    embed_candidates = filtered_embed_rows(args.embed_metadata, embed_labels, args.embed_root)

    ibia_labels = read_ibia_labels(args.ibia_metadata)
    ibia_candidates = ibia_rows(args.ibia_root, ibia_labels)

    if args.balanced_per_class > 0:
        embed_selected, ibia_selected = balanced_sample_rows(
            embed_rows=embed_candidates,
            ibia_rows_=ibia_candidates,
            per_class=args.balanced_per_class,
            target_ibia_per_class=args.target_ibia_per_class,
            seed=args.seed,
        )
    else:
        embed_selected = sample_rows(embed_candidates, args.embed_count, args.seed)
        ibia_selected = sample_rows(ibia_candidates, args.ibia_count, args.seed)

    embed_sample = assign_image_ids(embed_selected, "embed")
    ibia_sample = assign_image_ids(ibia_selected, "ibia")

    write_csv(args.output_dir / "benchmark_test_public.csv", embed_sample + ibia_sample, include_labels=False)
    write_combined_labels(args.output_dir / "benchmark_labels_private.csv", embed_sample + ibia_sample)

    print(f"Selected {len(embed_sample)} EMBED rows and {len(ibia_sample)} IBIA rows.")
    print(f"Wrote {len(embed_sample) + len(ibia_sample)} combined rows to {args.output_dir / 'benchmark_labels_private.csv'}")


if __name__ == "__main__":
    main()
