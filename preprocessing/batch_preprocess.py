"""
batch_preprocess.py
====================
Runs the full Module 1 preprocessing pipeline (skull strip -> N4 bias
correction -> normalize -> 2.5D multi-slice selection) over an ENTIRE
organized dataset (datasets/ADNI/{CN,MCI,AD}/*.nii.gz) and caches the
result as compressed .npz files.

Why this exists: preprocessing a single volume (especially N4 bias
correction) takes real time. Re-running it from raw NIfTI on every
training epoch would be far too slow. This script does it ONCE, saves
the result, and the training dataloader (Module 5) then just loads the
cached .npz files directly -- fast.

Features:
- Resumable: skips files that were already successfully processed
  (safe to Ctrl+C and rerun).
- Parallelized across CPU cores (preprocessing is CPU-bound).
- Writes a manifest.csv (filepath, label, processed_path, status) so
  the dataset loader has a clean index, and a failures.csv so you can
  inspect anything that errored without losing the whole run.
"""

import os
import sys
import glob
import argparse
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import preprocess_volume
from utils.logger import get_logger
from utils.config import cfg

logger = get_logger("batch_preprocess")


def find_dataset_files(dataset_root: str, classes=("CN", "MCI", "AD")):
    """Return list of (filepath, label) for every .nii/.nii.gz under
    dataset_root/<class>/."""
    items = []
    for cls in classes:
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(cls_dir, "*.nii*"))):
            items.append((fp, cls))
    return items


def _process_one(args):
    """Worker function (must be top-level for multiprocessing pickling)."""
    filepath, label, out_dir, n_slices, out_size, do_bias_correction = args
    subject_name = os.path.basename(filepath).replace(".nii.gz", "").replace(".nii", "")
    out_path = os.path.join(out_dir, label, f"{subject_name}.npz")

    if os.path.exists(out_path):
        return {"filepath": filepath, "label": label, "processed_path": out_path, "status": "skipped_existing"}

    try:
        result = preprocess_volume(
            filepath,
            do_skull_strip=True,
            do_bias_correction=do_bias_correction,
            do_registration=False,
            n_slices=n_slices,
            out_size=out_size,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            sagittal=result["sagittal"],
            coronal=result["coronal"],
            axial=result["axial"],
            label=label,
        )
        return {"filepath": filepath, "label": label, "processed_path": out_path, "status": "ok"}
    except Exception as e:
        return {
            "filepath": filepath, "label": label, "processed_path": None,
            "status": f"FAILED: {type(e).__name__}: {e}",
        }


def run_batch(dataset_root, out_dir, n_slices=5, out_size=224, workers=None,
              do_bias_correction=True, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    for cls in cfg.class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    items = find_dataset_files(dataset_root)
    if limit:
        items = items[:limit]
    logger.info(f"Found {len(items)} files to process under {dataset_root}")

    tasks = [(fp, label, out_dir, n_slices, out_size, do_bias_correction) for fp, label in items]
    workers = workers or max(1, cpu_count() - 1)
    logger.info(f"Processing with {workers} parallel workers...")

    results = []
    with Pool(workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks), desc="Preprocessing"):
            results.append(res)

    df = pd.DataFrame(results)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    df.to_csv(manifest_path, index=False)

    failures = df[df["status"].str.startswith("FAILED", na=False)]
    failures_path = os.path.join(out_dir, "failures.csv")
    failures.to_csv(failures_path, index=False)

    ok = (df["status"] == "ok").sum()
    skipped = (df["status"] == "skipped_existing").sum()
    failed = len(failures)

    logger.info(f"Done. Newly processed: {ok}, already cached: {skipped}, failed: {failed}")
    logger.info(f"Manifest: {manifest_path}")
    if failed:
        logger.warning(f"{failed} files failed -- see {failures_path}")

    logger.info("Class distribution in manifest (ok + skipped):")
    good = df[df["status"].isin(["ok", "skipped_existing"])]
    logger.info(good["label"].value_counts().to_dict())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess an organized ADNI/AIBL/OASIS dataset")
    parser.add_argument("--dataset_root", required=True, help="e.g. datasets/ADNI (must contain CN/MCI/AD subfolders)")
    parser.add_argument("--out_dir", required=True, help="e.g. datasets/processed/ADNI")
    parser.add_argument("--n_slices", type=int, default=cfg.n_slices_per_plane)
    parser.add_argument("--out_size", type=int, default=cfg.image_size)
    parser.add_argument("--workers", type=int, default=None, help="default: CPU count - 1")
    parser.add_argument("--skip_bias_correction", action="store_true", help="faster, lower quality -- for quick tests only")
    parser.add_argument("--limit", type=int, default=None, help="only process first N files (for testing)")
    args = parser.parse_args()

    run_batch(
        args.dataset_root, args.out_dir,
        n_slices=args.n_slices, out_size=args.out_size, workers=args.workers,
        do_bias_correction=not args.skip_bias_correction, limit=args.limit,
    )