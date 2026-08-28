"""
normalization.py
=================
Intensity normalization for structural MRI volumes.

Why this matters: different scanners/cohorts (ADNI vs AIBL vs OASIS) produce
voxel intensities on completely different numeric scales. Without
normalization, the model would need to learn scanner-specific intensity
statistics instead of anatomy -- directly hurting cross-cohort
generalization (the core claim of AD-HyFormer).

Method: percentile clipping (removes extreme outlier voxels, e.g. scanner
artifacts) followed by z-score normalization computed only over brain
voxels (intensity > 0), so background doesn't skew the statistics.
"""

import numpy as np


def clip_percentiles(volume: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Clip extreme intensity outliers using percentiles computed on
    non-zero (brain) voxels only."""
    brain_voxels = volume[volume > 0]
    if brain_voxels.size == 0:
        return volume
    lo, hi = np.percentile(brain_voxels, [lower, upper])
    return np.clip(volume, lo, hi)


def zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance normalization over brain voxels only.
    Background stays at 0."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / (std + eps)
    return normalized


def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]. Used right before saving to 8-bit slice images."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - vmin) / (vmax - vmin)


def normalize_volume(volume: np.ndarray, clip_pct=(0.5, 99.5)) -> np.ndarray:
    """Full normalization pipeline: clip -> z-score -> [0,1] rescale."""
    volume = clip_percentiles(volume, *clip_pct)
    volume = zscore_normalize(volume)
    volume = min_max_normalize(volume)
    return volume.astype(np.float32)
