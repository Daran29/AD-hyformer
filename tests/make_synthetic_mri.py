"""
make_synthetic_mri.py
======================
Generates a synthetic, brain-shaped NIfTI volume purely for smoke-testing
the preprocessing/model pipeline in an environment without access to real
ADNI/AIBL/OASIS data. NOT used for anything scientific -- swap in real
NIfTI files from your dataset for actual training.
"""

import numpy as np
import nibabel as nib
import os


def make_synthetic_brain_volume(shape=(96, 112, 96), seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij",
    )
    # ellipsoid "brain" mask
    dist = (xx / 0.8) ** 2 + (yy / 0.9) ** 2 + (zz / 0.8) ** 2
    brain_mask = (dist < 1.0).astype(np.float32)

    # smooth intensity texture + gyri-like noise inside the brain
    base_intensity = 600 + 150 * np.exp(-3 * dist)
    texture = rng.normal(0, 40, size=shape)
    from scipy.ndimage import gaussian_filter
    texture = gaussian_filter(texture, sigma=1.5)

    volume = (base_intensity + texture) * brain_mask

    # simulate a smooth multiplicative bias field (what N4 correction fixes)
    bias = 1.0 + 0.3 * (xx + yy + zz)
    volume = volume * bias

    volume[volume < 0] = 0
    return volume.astype(np.float32)


def save_as_nifti(volume: np.ndarray, out_path: str):
    img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(img, out_path)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "sample_data")
    os.makedirs(out_dir, exist_ok=True)
    for i, label in enumerate(["CN", "MCI", "AD"]):
        vol = make_synthetic_brain_volume(seed=i)
        out_path = os.path.join(out_dir, f"synthetic_{label}_001.nii.gz")
        save_as_nifti(vol, out_path)
        print(f"Saved {out_path}  shape={vol.shape}")
