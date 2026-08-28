"""
preprocess.py
=============
Top-level preprocessing pipeline for a single MRI file.

Pipeline: load -> skull strip -> N4 bias correction -> intensity
normalize -> (optional) rigid registration to reference -> multi-slice
(2.5D) selection -> resize.

Supports .nii / .nii.gz (via nibabel) and pre-extracted .png / .jpg
slice images (loaded directly, skipping the 3D-only steps).
"""

import os
import sys
import numpy as np
import nibabel as nib
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.registration import skull_strip, bias_field_correction, rigid_register
from preprocessing.normalization import normalize_volume
from preprocessing.slice_selection import build_multiplanar_2_5d
from utils.logger import get_logger
from utils.config import cfg

logger = get_logger("preprocess")

NIFTI_EXTS = (".nii", ".nii.gz")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_volume(filepath: str) -> np.ndarray:
    """Load a 3D MRI volume from a NIfTI file."""
    img = nib.load(filepath)
    volume = img.get_fdata().astype(np.float32)
    return volume


def preprocess_volume(
    filepath: str,
    reference_volume: np.ndarray = None,
    do_skull_strip: bool = True,
    do_bias_correction: bool = True,
    do_registration: bool = False,
    n_slices: int = None,
    out_size: int = None,
) -> dict:
    """
    Run the full preprocessing pipeline on one NIfTI volume.

    Parameters
    ----------
    filepath : path to .nii / .nii.gz file
    reference_volume : if do_registration=True, the fixed/reference volume
    do_skull_strip, do_bias_correction, do_registration : toggle stages
        (bias correction / registration are expensive -- disable for quick
        smoke tests, enable for the real training pipeline)
    n_slices, out_size : override cfg defaults

    Returns
    -------
    dict: {"sagittal": (n,H,W), "coronal": (n,H,W), "axial": (n,H,W)}
    """
    n_slices = n_slices or cfg.n_slices_per_plane
    out_size = out_size or cfg.image_size

    logger.info(f"Loading {filepath}")
    volume = load_volume(filepath)

    if do_skull_strip:
        logger.info("Skull stripping...")
        volume = skull_strip(volume)

    if do_bias_correction:
        logger.info("N4 bias field correction...")
        volume = bias_field_correction(volume)

    if do_registration and reference_volume is not None:
        logger.info("Rigid registration to reference...")
        volume = rigid_register(volume, reference_volume)

    logger.info("Intensity normalization...")
    volume = normalize_volume(volume, cfg.intensity_clip_percentiles)

    logger.info(f"Selecting {n_slices} slices per plane (2.5D)...")
    slices = build_multiplanar_2_5d(volume, n_slices=n_slices, out_size=out_size)

    return slices


def preprocess_image_folder(filepaths, out_size=None) -> np.ndarray:
    """For datasets already distributed as 2D slice images (png/jpg)
    rather than 3D NIfTI volumes: load, grayscale, resize, normalize."""
    out_size = out_size or cfg.image_size
    slices = []
    for fp in filepaths:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        img = normalize_volume(img)
        slices.append(img)
    return np.stack(slices, axis=0)


def is_nifti(filepath: str) -> bool:
    return filepath.lower().endswith(NIFTI_EXTS)


def is_image(filepath: str) -> bool:
    return filepath.lower().endswith(IMAGE_EXTS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a single MRI volume")
    parser.add_argument("--input", required=True, help="Path to .nii/.nii.gz file")
    parser.add_argument("--n_slices", type=int, default=cfg.n_slices_per_plane)
    parser.add_argument("--out_size", type=int, default=cfg.image_size)
    parser.add_argument("--skip_bias_correction", action="store_true")
    args = parser.parse_args()

    result = preprocess_volume(
        args.input,
        do_bias_correction=not args.skip_bias_correction,
        n_slices=args.n_slices,
        out_size=args.out_size,
    )
    for plane, arr in result.items():
        logger.info(f"{plane}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
