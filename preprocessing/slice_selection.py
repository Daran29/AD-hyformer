"""
slice_selection.py
===================
Multi-slice (2.5D) selection: instead of the base paper's single
mid-sagittal slice, AD-HyFormer extracts N informative slices from each
of the three anatomical planes (sagittal, coronal, axial), so the model
sees richer 3D structural context while staying computationally cheap
(no full 3D CNN required).

"Informative" is defined as: slices centered on the volume's midline
along that axis, restricted to a window where brain-tissue content
(non-zero voxel fraction) is high -- this avoids selecting near-empty
edge slices.
"""

import numpy as np
import cv2
from typing import List


PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def _tissue_fraction(slice_2d: np.ndarray) -> float:
    return float((slice_2d > 0).mean())


def select_informative_slice_indices(
    volume: np.ndarray, axis: int, n_slices: int, min_tissue_fraction: float = 0.05
) -> List[int]:
    """Pick n_slices indices along `axis`, centered on the midline,
    skipping slices with too little brain tissue."""
    size = volume.shape[axis]
    mid = size // 2

    # Rank candidate indices near the midline by tissue content
    window = max(n_slices * 4, 20)
    candidates = list(range(max(0, mid - window), min(size, mid + window)))

    scored = []
    for idx in candidates:
        sl = np.take(volume, idx, axis=axis)
        frac = _tissue_fraction(sl)
        if frac >= min_tissue_fraction:
            scored.append((abs(idx - mid), idx))  # distance to midline, index

    if len(scored) < n_slices:
        # fallback: not enough tissue-rich slices found, just take midline block
        start = max(0, mid - n_slices // 2)
        return list(range(start, min(size, start + n_slices)))

    scored.sort(key=lambda x: x[0])
    chosen = sorted(idx for _, idx in scored[:n_slices])
    return chosen


def extract_slices(volume: np.ndarray, axis: int, indices: List[int], out_size: int) -> np.ndarray:
    """Extract and resize slices along `axis` at given indices ->
    array of shape (n_slices, out_size, out_size)."""
    slices = []
    for idx in indices:
        sl = np.take(volume, idx, axis=axis).astype(np.float32)
        resized = cv2.resize(sl, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        slices.append(resized)
    return np.stack(slices, axis=0)


def build_multiplanar_2_5d(volume: np.ndarray, n_slices: int = 5, out_size: int = 224) -> dict:
    """Build the full 2.5D representation used by AD-HyFormer:
    n_slices from each of sagittal, coronal, axial planes.

    Returns
    -------
    dict with keys "sagittal", "coronal", "axial", each of shape
    (n_slices, out_size, out_size).
    """
    result = {}
    for plane, axis in PLANE_AXIS.items():
        indices = select_informative_slice_indices(volume, axis, n_slices)
        result[plane] = extract_slices(volume, axis, indices, out_size)
    return result
