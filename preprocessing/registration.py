"""
registration.py
================
Skull stripping, N4 bias field correction, and rigid registration to a
reference template -- implemented with SimpleITK (the standard toolkit
used in medical imaging research, same engine ANTs/N4ITK is built on).

NOTE on skull stripping: production pipelines typically use a dedicated
deep model (HD-BET, FSL BET, ROBEX). Those require large pretrained
weights that aren't available in this environment, so we implement an
Otsu-threshold + largest-connected-component brain mask, which is the
classical (pre-deep-learning) approach and works reasonably well as a
drop-in you can later swap for HD-BET without changing the pipeline's
interface (`skull_strip(volume) -> masked_volume`).
"""

import numpy as np
import SimpleITK as sitk


def to_sitk(volume: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing(spacing)
    return img


def skull_strip(volume: np.ndarray) -> np.ndarray:
    """Otsu-threshold based brain extraction + largest connected component."""
    img = to_sitk(volume)

    # Otsu thresholding separates brain tissue from background/skull rim
    otsu = sitk.OtsuThreshold(img, 0, 1, 200)

    # Morphological cleanup: fill holes, keep largest connected component
    filled = sitk.BinaryFillhole(otsu)
    cc = sitk.ConnectedComponent(filled)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    brain_mask = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)

    # Smooth mask edges slightly
    brain_mask = sitk.BinaryMorphologicalClosing(brain_mask, [2, 2, 2])

    mask_arr = sitk.GetArrayFromImage(brain_mask).astype(np.float32)
    return volume * mask_arr


def bias_field_correction(volume: np.ndarray, shrink_factor: int = 2) -> np.ndarray:
    """N4 bias field correction (the standard MRI intensity-inhomogeneity
    correction algorithm used across neuroimaging pipelines)."""
    img = to_sitk(volume)
    img = sitk.Cast(img, sitk.sitkFloat32)

    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)

    # Shrink for speed, correct, then the field is applied at full res
    shrunk_img = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    shrunk_mask = sitk.Shrink(mask_img, [shrink_factor] * img.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 20, 20])
    _ = corrector.Execute(shrunk_img, shrunk_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_img = img / sitk.Exp(log_bias_field)

    return sitk.GetArrayFromImage(corrected_img)


def rigid_register(volume: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigid-body registration of `volume` onto `reference` space
    (e.g. an MNI152 template or an ADNI reference subject), so all
    subjects/cohorts share the same anatomical orientation and size."""
    fixed = to_sitk(reference)
    moving = to_sitk(volume)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return sitk.GetArrayFromImage(resampled)