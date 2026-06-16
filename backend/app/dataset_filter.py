from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_embed_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Apply the initial EMBED mammogram selection criteria."""

    metadata = pd.read_csv(metadata_csv)
    spot_mag = metadata.get("spot_mag")
    if spot_mag is None:
        spot_mag_blank = True
    else:
        spot_mag_blank = spot_mag.isna() | (spot_mag.astype(str).str.strip() == "")

    return metadata[
        (metadata["FinalImageType"].astype(str) == "2D")
        & (metadata["ViewPosition"].astype(str).isin(["CC", "MLO"]))
        & spot_mag_blank
        & (pd.to_numeric(metadata["XRayTubeCurrent"], errors="coerce") >= 100)
        & (pd.to_numeric(metadata["BitsStored"], errors="coerce") == 12.0)
        & (metadata["PatientSex"].astype(str) == "F")
        & (metadata["BreastImplantPresent"].astype(str).str.upper() == "NO")
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the private benchmark subset manifest.")
    parser.add_argument("--metadata", required=True, type=Path, help="Path to full EMBED metadata CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Path for filtered manifest CSV.")
    args = parser.parse_args()

    filtered = filter_embed_metadata(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    print(f"Wrote {len(filtered)} selected rows to {args.output}")


if __name__ == "__main__":
    main()
