import SimpleITK as sitk


def bias_correct(
    input: sitk.Image,
    mask: sitk.Image = None,
    shrink_factor: int = 4,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
) -> sitk.Image:
    """Perform N4 bias correction on MRI

    Note:
    - if no output_path is provided the input will be overwritten
    - if no mask is provided it will be generated using Otsu-thresholding
    """
    if not isinstance(mask, sitk.Image):
        mask = sitk.OtsuThreshold(input, 0, 1, 200)

    input = sitk.Cast(input, sitk.sitkFloat32)
    image = sitk.Shrink(
        sitk.Cast(input, sitk.sitkFloat32), [shrink_factor] * input.GetDimension()
    )
    mask = sitk.Shrink(mask, [shrink_factor] * input.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)

    corrector.Execute(image, mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input)
    corrected_full_resolution: sitk.Image = input / sitk.Exp(log_bias_field)
    return corrected_full_resolution


def scale_clamp_ct(img: sitk.Image) -> sitk.Image:
    """Prepare CT images: median -> clamp to [-1100,3100] -> scale to [0,255]"""
    # median filter for salt and pepper noise
    img = sitk.Median(img, radius=[1] * img.GetDimension())
    # range clamped to [-1100, 3100]
    img = sitk.Clamp(-1100, 3100)
    # and scaled to [0, 255]
    img = sitk.ShiftScale(img, shift=1100, scale=255.0 / (1100.0 + 3100.0))
    return img


def unscale_ct(img: sitk.Image) -> sitk.Image:
    """Invert 'scale_clamp_ct' operation, except for clamping"""
    img = (1100.0 + 3100.0) / 255.0 * img - 1100
    return img
