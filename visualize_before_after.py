"""Visualize one MRI before and after AD-HyFormer preprocessing.

Place this file in the root of the AD-hyformer repository, change MRI_FILE
below, and run: python visualize_before_after.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.preprocess import load_volume, preprocess_volume


# ======================== CHANGE ONLY THIS LINE ========================
MRI_FILE = r"path/to/your/mri_file.nii.gz"
# ======================================================================

# Set False if N4 bias correction is too slow for a quick preview.
USE_BIAS_CORRECTION = True
N_SLICES = 5
IMAGE_SIZE = 224
OUTPUT_FILE = "before_after_preprocessing.png"


def display_normalize(image: np.ndarray) -> np.ndarray:
    """Scale a slice to [0, 1] for correct visualization."""
    image = np.asarray(image, dtype=np.float32)
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def main() -> None:
    path = Path(MRI_FILE)
    if not path.is_file():
        raise FileNotFoundError(
            f"MRI file not found: {path}\n"
            "Change MRI_FILE at the top of this script to a valid .nii or .nii.gz file."
        )

    print(f"Loading: {path}")
    raw = load_volume(str(path))

    print("Running preprocessing...")
    processed = preprocess_volume(
        str(path),
        do_skull_strip=True,
        do_bias_correction=USE_BIAS_CORRECTION,
        do_registration=False,
        n_slices=N_SLICES,
        out_size=IMAGE_SIZE,
    )

    raw_middle = {
        "Sagittal": raw[raw.shape[0] // 2, :, :],
        "Coronal": raw[:, raw.shape[1] // 2, :],
        "Axial": raw[:, :, raw.shape[2] // 2],
    }
    processed_middle = {
        "Sagittal": processed["sagittal"][N_SLICES // 2],
        "Coronal": processed["coronal"][N_SLICES // 2],
        "Axial": processed["axial"][N_SLICES // 2],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    planes = ("Sagittal", "Coronal", "Axial")

    for column, plane in enumerate(planes):
        axes[0, column].imshow(
            np.rot90(display_normalize(raw_middle[plane])), cmap="gray"
        )
        axes[0, column].set_title(f"Before: {plane}", fontsize=13)
        axes[0, column].axis("off")

        axes[1, column].imshow(
            np.rot90(processed_middle[plane]), cmap="gray", vmin=0, vmax=1
        )
        axes[1, column].set_title(f"After: {plane}", fontsize=13)
        axes[1, column].axis("off")

    fig.suptitle("MRI Before and After Preprocessing", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"Input: {path.name} | Raw shape: {raw.shape} | Output: {IMAGE_SIZE} x {IMAGE_SIZE}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved comparison to: {Path(OUTPUT_FILE).resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
