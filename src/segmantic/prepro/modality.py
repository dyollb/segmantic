import itk
import numpy as np

from .core import ImageAnyd, NdarrayImage, as_array


def bias_correct(
    image: ImageAnyd,
    mask: ImageAnyd = None,
    shrink_factor: int = 4,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
) -> ImageAnyd:
    """Perform N4 bias correction on MRI

    Note:
    - if no mask is provided it will be generated using Otsu-thresholding
    """
    if mask is None:
        mask = itk.otsu_threshold_image_filter(
            image, outside_value=0, mask_value=1, number_of_histogram_bins=200
        )

    image = image.astype(itk.F)
    mask = mask.astype(itk.UC)
    image_small = itk.shrink_image_filter(
        image, shrink_factors=[shrink_factor] * image.GetImageDimension()
    )
    mask_small = itk.shrink_image_filter(
        mask, shrink_factors=[shrink_factor] * image.GetImageDimension()
    )

    corrector = itk.N4BiasFieldCorrectionImageFilter.New(image_small, mask_small)
    corrector.SetNumberOfFittingLevels(num_fitting_levels)
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)
    corrector.Update()

    # prevent ReconstructBiasField from setting all regions to a smaller RequestedRegion
    # https://github.com/InsightSoftwareConsortium/ITK/pull/3477
    image.SetRequestedRegion(image.GetBufferedRegion())

    # reconstruct at full resoluton
    full_res_corrector = itk.N4BiasFieldCorrectionImageFilter.New(image, mask)
    log_bias_field = full_res_corrector.ReconstructBiasField(
        corrector.GetLogBiasFieldControlPointLattice()
    )

    corrected_image_np = itk.array_view_from_image(image) / np.exp(
        itk.array_view_from_image(log_bias_field)
    )
    corrected_image_full_resolution = itk.image_from_array(corrected_image_np)
    corrected_image_full_resolution.CopyInformation(image)
    return corrected_image_full_resolution


def scale_clamp_ct(img: ImageAnyd) -> ImageAnyd:
    """Prepare CT images: median -> clamp to [-1100,3100] -> scale to [0,255]"""
    # median filter for salt and pepper noise
    img = itk.median_image_filter(img, radius=1)
    # range clamped to [-1100, 3100]
    img_view = itk.array_view_from_image(img)
    np.clip(img_view, a_min=-1100, a_max=3100, out=img_view)
    # and scaled to [0, 255]
    img_view += 1100
    np.multiply(img_view, 255.0 / (1100.0 + 3100.0), out=img_view, casting="unsafe")
    return img


def unscale_ct(img: NdarrayImage) -> NdarrayImage:
    """Invert 'scale_clamp_ct' operation, except for clamping"""
    img_view = as_array(img)
    np.multiply(img_view, (1100.0 + 3100.0) / 255.0, out=img_view, casting="unsafe")
    img_view -= 1100
    return img
