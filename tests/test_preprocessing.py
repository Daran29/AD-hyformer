"""
test_preprocessing.py
======================
Unit tests for Module 1 (MRI preprocessing). Run with: pytest tests/test_preprocessing.py -v
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.normalization import clip_percentiles, zscore_normalize, min_max_normalize, normalize_volume
from preprocessing.slice_selection import select_informative_slice_indices, build_multiplanar_2_5d
from preprocessing.registration import skull_strip
from tests.make_synthetic_mri import make_synthetic_brain_volume

VOL = make_synthetic_brain_volume(shape=(64, 72, 64), seed=1)


def test_clip_percentiles_reduces_range():
    clipped = clip_percentiles(VOL, 1, 99)
    assert clipped.max() <= VOL.max()
    assert clipped.shape == VOL.shape


def test_zscore_normalize_zero_mean_on_brain():
    z = zscore_normalize(VOL)
    brain = z[VOL > 0]
    assert abs(brain.mean()) < 1e-3
    assert abs(brain.std() - 1.0) < 1e-2


def test_min_max_normalize_bounds():
    m = min_max_normalize(VOL)
    assert m.min() >= 0.0 and m.max() <= 1.0 + 1e-6


def test_normalize_volume_pipeline():
    out = normalize_volume(VOL)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    assert out.shape == VOL.shape


def test_skull_strip_removes_background_signal():
    stripped = skull_strip(VOL)
    # background corners should be zero after stripping
    assert stripped[0, 0, 0] == 0
    assert stripped.shape == VOL.shape


def test_slice_selection_returns_correct_count_and_no_duplicates():
    indices = select_informative_slice_indices(VOL, axis=0, n_slices=5)
    assert len(indices) == 5
    assert len(set(indices)) == 5  # no duplicate slice indices
    assert all(0 <= i < VOL.shape[0] for i in indices)


def test_build_multiplanar_2_5d_shapes():
    result = build_multiplanar_2_5d(VOL, n_slices=5, out_size=224)
    assert set(result.keys()) == {"sagittal", "coronal", "axial"}
    for plane, arr in result.items():
        assert arr.shape == (5, 224, 224)
        assert not np.isnan(arr).any()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
