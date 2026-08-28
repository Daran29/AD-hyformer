"""
organize_adni_dataset.py
=========================
Reads an ADNI diagnosis CSV (DXSUM_PDXCONV_*.csv or ADNIMERGE_*.csv --
both column layouts are supported, see _extract_diagnosis_map below) and
copies each subject's converted NIfTI volume into

    datasets/ADNI/<CN|MCI|AD>/<subject_id>_<original_filename>.nii.gz

so the folder layout matches exactly what preprocessing/preprocess.py
and the dataset loader (Module 2+) expect.

ADNI diagnosis coding reference (handled automatically below):
  DXSUM_PDXCONV (ADNI1 legacy):  DXCURREN   1=Normal(CN) 2=MCI 3=AD
  DXSUM_PDXCONV (ADNI2/GO/3/4):  DIAGNOSIS  1=CN         2=MCI 3=Dementia(AD)
  ADNIMERGE:                     DX         'CN' / 'MCI' / 'Dementia'
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

logger = get_logger("organize_adni_dataset")

# Maps raw diagnosis codes/strings (from any of ADNI's CSV formats) -> our 3 classes
_CODE_MAP = {
    # DXCURREN / DIAGNOSIS numeric codes
    "1": "CN", "1.0": "CN",
    "2": "MCI", "2.0": "MCI",
    "3": "AD", "3.0": "AD",
    # ADNIMERGE string labels
    "CN": "CN", "NL": "CN",
    "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI",
    "AD": "AD", "DEMENTIA": "AD",
}

_DIAGNOSIS_COLUMNS = ["DIAGNOSIS", "DXCURREN", "DX"]  # checked in this order
_SUBJECT_ID_PATTERN = re.compile(r"\d{3}_S_\d{4}")


def _extract_diagnosis_map(csv_path: str) -> Dict[str, str]:
    """Build {subject_id: 'CN'|'MCI'|'AD'} from an ADNI clinical CSV.

    If a subject has multiple visits/rows, the most recent non-null
    diagnosis is used.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    id_col = next((c for c in ["PTID", "SUBJECT_ID", "RID"] if c in df.columns), None)
    dx_col = next((c for c in _DIAGNOSIS_COLUMNS if c in df.columns), None)

    if id_col is None or dx_col is None:
        raise ValueError(
            f"Could not find subject-ID / diagnosis columns in {csv_path}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected one of {['PTID', 'SUBJECT_ID', 'RID']} and one of {_DIAGNOSIS_COLUMNS}."
        )

    # keep the last non-null diagnosis per subject (most recent visit)
    df = df[[id_col, dx_col]].dropna()
    df[dx_col] = df[dx_col].astype(str).str.strip().str.upper()

    diagnosis_map = {}
    for subject_id, group in df.groupby(id_col):
        raw_label = group[dx_col].iloc[-1]
        mapped = _CODE_MAP.get(raw_label)
        if mapped:
            diagnosis_map[str(subject_id)] = mapped
        else:
            logger.warning(f"Unrecognized diagnosis code '{raw_label}' for subject {subject_id}, skipping.")

    logger.info(f"Parsed diagnoses for {len(diagnosis_map)} subjects from {os.path.basename(csv_path)}")
    return diagnosis_map


def _subject_id_from_filename(filepath: str) -> str:
    match = _SUBJECT_ID_PATTERN.search(os.path.basename(filepath))
    return match.group(0) if match else None


def organize(nifti_root: str, diagnosis_csv: str, output_root: str, copy: bool = True):
    """
    Parameters
    ----------
    nifti_root : folder of converted .nii.gz files (output of dicom_to_nifti.py),
                 searched recursively
    diagnosis_csv : path to DXSUM_PDXCONV_*.csv or ADNIMERGE_*.csv
    output_root : e.g. datasets/ADNI  -- CN/MCI/AD subfolders created inside
    copy : True = copy files (safe, default). False = move (frees disk space).
    """
    diagnosis_map = _extract_diagnosis_map(diagnosis_csv)

    for cls in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

    stats = {"CN": 0, "MCI": 0, "AD": 0, "unmatched": 0}

    for dirpath, _, filenames in os.walk(nifti_root):
        for fname in filenames:
            if not fname.endswith((".nii", ".nii.gz")):
                continue
            filepath = os.path.join(dirpath, fname)
            subject_id = _subject_id_from_filename(filepath)

            label = diagnosis_map.get(subject_id) if subject_id else None
            if label is None:
                logger.warning(f"No diagnosis found for {fname} (subject={subject_id}), skipping.")
                stats["unmatched"] += 1
                continue

            dest = os.path.join(output_root, label, fname)
            if copy:
                shutil.copy2(filepath, dest)
            else:
                shutil.move(filepath, dest)
            stats[label] += 1

    logger.info(
        f"Organized dataset -> CN: {stats['CN']}, MCI: {stats['MCI']}, "
        f"AD: {stats['AD']}, unmatched/skipped: {stats['unmatched']}"
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort converted ADNI NIfTI files into CN/MCI/AD folders")
    parser.add_argument("--nifti_root", required=True, help="Folder of converted .nii.gz files")
    parser.add_argument("--diagnosis_csv", required=True, help="Path to DXSUM_PDXCONV or ADNIMERGE csv")
    parser.add_argument("--output_root", required=True, help="e.g. datasets/ADNI")
    parser.add_argument("--move", action="store_true", help="Move instead of copy files")
    args = parser.parse_args()

    organize(args.nifti_root, args.diagnosis_csv, args.output_root, copy=not args.move)