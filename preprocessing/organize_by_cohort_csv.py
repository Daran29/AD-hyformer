"""
organize_by_cohort_csv.py
==========================
Organizes converted NIfTI files into CN/MCI/AD folders using a
pre-finalized cohort CSV (columns: "Subject ID", "AssignedDiagnosis",
"Image ID", ...) rather than the raw DXSUM file.

This is the preferred organizer when you've already built a
one-scan-per-subject cohort selection (e.g. via IDA Advanced Search +
a priority-ranking step) -- it matches strictly on Image ID, which is
unambiguous even if a subject has multiple scans in your download.

ADNI filenames (including what dicom_to_nifti.py in this project
produces) contain the image ID as "I<digits>", e.g.:
    002_S_0685_I241350.nii.gz  -> Image ID = 241350
"""

import os
import re
import shutil
import argparse
import pandas as pd
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("organize_by_cohort_csv")

_DIAGNOSIS_MAP = {"NC": "CN", "CN": "CN", "MCI": "MCI", "AD": "AD"}
_IMAGE_ID_PATTERN = re.compile(r"[Ii](\d+)")


def load_cohort_map(csv_path: str) -> Dict[str, str]:
    """Build {image_id (str): 'CN'|'MCI'|'AD'} from the cohort CSV."""
    df = pd.read_csv(csv_path, dtype={"Image ID": str})

    if "Image ID" not in df.columns or "AssignedDiagnosis" not in df.columns:
        raise ValueError(
            f"Expected columns 'Image ID' and 'AssignedDiagnosis' in {csv_path}, "
            f"found: {list(df.columns)}"
        )

    dup = df["Image ID"].duplicated().sum()
    if dup:
        logger.warning(f"{dup} duplicate Image IDs found in cohort CSV -- keeping the first occurrence.")
        df = df.drop_duplicates(subset="Image ID", keep="first")

    image_id_map = {}
    unrecognized = set()
    for _, row in df.iterrows():
        label = _DIAGNOSIS_MAP.get(str(row["AssignedDiagnosis"]).strip().upper())
        if label is None:
            unrecognized.add(row["AssignedDiagnosis"])
            continue
        image_id_map[str(row["Image ID"]).strip()] = label

    if unrecognized:
        logger.warning(f"Unrecognized AssignedDiagnosis values (skipped): {unrecognized}")

    logger.info(f"Loaded {len(image_id_map)} labeled subjects from cohort CSV.")
    return image_id_map


def extract_image_id(filename: str) -> str:
    """Pull the numeric Image ID out of an ADNI-style filename."""
    match = _IMAGE_ID_PATTERN.search(filename)
    return match.group(1) if match else None


def organize(nifti_root: str, cohort_csv: str, output_root: str, copy: bool = True):
    image_id_map = load_cohort_map(cohort_csv)

    for cls in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

    stats = {"CN": 0, "MCI": 0, "AD": 0, "unmatched": 0}
    matched_image_ids = set()

    for dirpath, _, filenames in os.walk(nifti_root):
        for fname in filenames:
            if not fname.endswith((".nii", ".nii.gz")):
                continue
            filepath = os.path.join(dirpath, fname)
            image_id = extract_image_id(fname)

            label = image_id_map.get(image_id) if image_id else None
            if label is None:
                logger.warning(f"No cohort match for {fname} (image_id={image_id}), skipping.")
                stats["unmatched"] += 1
                continue

            dest = os.path.join(output_root, label, fname)
            (shutil.copy2 if copy else shutil.move)(filepath, dest)
            stats[label] += 1
            matched_image_ids.add(image_id)

    missing = set(image_id_map.keys()) - matched_image_ids
    logger.info(
        f"Organized -> CN: {stats['CN']}, MCI: {stats['MCI']}, AD: {stats['AD']}, "
        f"unmatched files: {stats['unmatched']}, cohort IDs not found on disk: {len(missing)}"
    )
    if missing:
        logger.info(f"First 10 missing Image IDs (not downloaded/converted yet?): {list(missing)[:10]}")
    return stats, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort converted ADNI NIfTI files using final_cohort.csv")
    parser.add_argument("--nifti_root", required=True, help="Folder of converted .nii.gz files")
    parser.add_argument("--cohort_csv", required=True, help="Path to final_cohort.csv")
    parser.add_argument("--output_root", required=True, help="e.g. datasets/ADNI")
    parser.add_argument("--move", action="store_true", help="Move instead of copy files")
    args = parser.parse_args()

    organize(args.nifti_root, args.cohort_csv, args.output_root, copy=not args.move)