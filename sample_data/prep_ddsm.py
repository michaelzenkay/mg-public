"""
prep_ddsm.py — Convert CBIS-DDSM DICOMs to PNGs and build a mg-public CSV.

Usage:
    python prep_ddsm.py --ddsm_dir C:/ddsm --out_dir C:/ddsm_png --out_csv ddsm_mg.csv

Requirements (in addition to mg-public's requirements.txt):
    pip install pydicom
"""

import argparse
import re
import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    raise ImportError("Install pydicom:  pip install pydicom")


LATERALITY_MAP = {"LEFT": "L", "RIGHT": "R"}

# Folder name pattern: Calc-Test_P_00038_LEFT_CC  (no trailing _N)
FOLDER_RE = re.compile(r"^(?:Calc|Mass)-(?:Test|Train)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)$")


def find_dicom(case_dir: Path) -> Path | None:
    """Walk into <date>/<series>/1-1.dcm inside a full-mammogram case folder."""
    for date_dir in case_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for series_dir in date_dir.iterdir():
            if not series_dir.is_dir():
                continue
            dcm = series_dir / "1-1.dcm"
            if dcm.exists():
                return dcm
    return None


def dcm_to_png(dcm_path: Path, png_path: Path) -> None:
    """Convert DICOM to 16-bit PNG matching dcm2img +on2 --min-max window.

    Pipeline mirrors dcm2img:
      1. Apply modality LUT (RescaleSlope / RescaleIntercept)
      2. Handle MONOCHROME1 polarity inversion
      3. Min-max window: normalise to full 16-bit range
      4. Write 16-bit grayscale PNG
    """
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    # 1. Modality LUT — apply rescale slope/intercept (usually 1/0 for mammo,
    #    but dcm2img does this before windowing so we match it)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # 2. MONOCHROME1 inversion (bright background → invert so tissue is bright)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # 3. Min-max window → 16-bit output  (matches --min-max window + +on2)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 65535.0

    # 4. Save as 16-bit PNG (PIL mode "I;16" writes little-endian uint16)
    img = Image.fromarray(arr.astype(np.uint16), mode="I;16")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(png_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddsm_dir", required=True, help="Root of C:/ddsm")
    parser.add_argument("--out_dir", required=True, help="Where to save PNGs")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--split_group", default="test")
    parser.add_argument(
        "--followup_months",
        type=float,
        default=24.0,
        help="Placeholder followup months for NOBC rows (converted to years_to_last_followup)",
    )
    args = parser.parse_args()

    ddsm_dir = Path(args.ddsm_dir)
    out_dir = Path(args.out_dir)

    rows = []
    skipped = []

    for folder in sorted(ddsm_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = FOLDER_RE.match(folder.name)
        if not m:
            # Folder has trailing _N  → ROI mask, skip
            continue

        pid, lat_long, view = m.group(1), m.group(2), m.group(3)
        laterality = LATERALITY_MAP[lat_long]

        dcm_path = find_dicom(folder)
        if dcm_path is None:
            print(f"  [WARN] No 1-1.dcm found in {folder.name}, skipping")
            skipped.append(folder.name)
            continue

        # One exam per patient in DDSM — all views share the same pid_acc
        pid_acc = f"{pid}_1"

        png_rel = Path(pid) / f"{pid}_{laterality}{view}.png"
        png_path = out_dir / png_rel

        print(f"  Converting {folder.name} -> {png_path.name} ...", end=" ", flush=True)
        dcm_to_png(dcm_path, png_path)
        print("done")

        rows.append(
            {
                "patient_id": pid,
                "exam_id": pid_acc,
                "laterality": laterality,
                "view": view,
                "file_path": str(png_path),
                "years_to_cancer": 100,
                "years_to_last_followup": args.followup_months / 12.0,
                "split_group": args.split_group,
            }
        )

    # Write MIRAI-format CSV
    fieldnames = ["patient_id", "exam_id", "laterality", "view", "file_path",
                  "years_to_cancer", "years_to_last_followup", "split_group"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out_csv}")

    # Summary
    pids = sorted({r["patient_id"] for r in rows})
    print(f"Patients: {len(pids)}  ({', '.join(pids)})")

    complete = []
    for pid in pids:
        views = {r["laterality"] + r["view"] for r in rows if r["patient_id"] == pid}
        has_all_4 = {"LCC", "LMLO", "RCC", "RMLO"}.issubset(views)
        status = "complete" if has_all_4 else f"incomplete ({', '.join(sorted(views))})"
        print(f"  {pid}: {status}")
        if has_all_4:
            complete.append(pid)

    print(f"\n{len(complete)}/{len(pids)} patients have all 4 views (LCC, LMLO, RCC, RMLO)")
    if skipped:
        print(f"Skipped folders: {skipped}")


if __name__ == "__main__":
    main()
