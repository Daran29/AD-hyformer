"""
dicom_to_nifti.py
==================
Converts ADNI's downloaded DICOM series into .nii.gz volumes using
dcm2niix (the standard, widely-validated DICOM->NIfTI converter used
across neuroimaging pipelines -- same tool used internally by fMRIPrep,
HeuDiConv, etc.).

ADNI's IDA download structure (per subject) typically looks like:

    ADNI/
      002_S_0685/
        MPRAGE/
          2006-04-18_08_14_32.0/
            I45108/
              ADNI_002_S_0685_MR_MPRAGE_..._I45108.dcm
              ADNI_002_S_0685_MR_MPRAGE_..._I45108.dcm
              ... (one .dcm per slice)

This script walks the downloaded tree, finds every leaf folder that
contains DICOM (.dcm) files, and converts each one into a single
.nii.gz file.
"""

import os
import subprocess
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("dicom_to_nifti")


def find_dicom_series_dirs(root_dir: str):
    """Find every folder that directly contains .dcm files (a 'series')."""
    series_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.lower().endswith(".dcm") for f in filenames):
            series_dirs.append(dirpath)
    return series_dirs


def convert_series(series_dir: str, out_dir: str, out_filename: str) -> str:
    """Convert one DICOM series folder into a single .nii.gz file.
    Returns the path to the produced file, or None if conversion failed.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",              # gzip output -> .nii.gz
        "-f", out_filename,     # output filename (no extension)
        "-o", out_dir,          # output directory
        "-b", "n",               # don't emit the .json sidecar (keeps things simple)
        series_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"dcm2niix failed for {series_dir}: {result.stderr.strip()}")
        return None

    produced = os.path.join(out_dir, out_filename + ".nii.gz")
    if os.path.exists(produced):
        return produced

    # dcm2niix sometimes appends a suffix (e.g. multi-echo series) -- grab
    # the first matching file if the exact name wasn't produced.
    candidates = list(Path(out_dir).glob(f"{out_filename}*.nii.gz"))
    return str(candidates[0]) if candidates else None


def subject_id_from_path(series_dir: str) -> str:
    """Extract ADNI subject ID (format 003_S_1234) from a series path."""
    import re
    match = re.search(r"\d{3}_S_\d{4}", series_dir)
    return match.group(0) if match else "UNKNOWN_SUBJECT"


def convert_all(input_root: str, output_root: str):
    """Convert every DICOM series found under input_root into
    output_root/<subject_id>/<series_folder_name>.nii.gz
    """
    series_dirs = find_dicom_series_dirs(input_root)
    logger.info(f"Found {len(series_dirs)} DICOM series under {input_root}")

    converted = []
    for series_dir in series_dirs:
        subject_id = subject_id_from_path(series_dir)
        series_name = os.path.basename(series_dir)
        out_dir = os.path.join(output_root, subject_id)
        out_filename = f"{subject_id}_{series_name}"

        nifti_path = convert_series(series_dir, out_dir, out_filename)
        if nifti_path:
            converted.append((subject_id, nifti_path))
            logger.info(f"Converted: {series_dir} -> {nifti_path}")
        else:
            logger.warning(f"Skipped (conversion failed): {series_dir}")

    logger.info(f"Done. Converted {len(converted)}/{len(series_dirs)} series.")
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADNI DICOM downloads to NIfTI")
    parser.add_argument("--input", required=True, help="Root folder of downloaded DICOM data")
    parser.add_argument("--output", required=True, help="Folder to write converted .nii.gz files")
    args = parser.parse_args()
    convert_all(args.input, args.output)