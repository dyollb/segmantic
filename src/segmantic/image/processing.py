import math
from typing import Any, List, Optional, Sequence, Union

import SimpleITK as sitk

_COLLAPSE_STRATEGY_SUBMATRIX = 2


def make_image(
    shape: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    value: Union[int, float] = 0,
    pixel_type: Any = sitk.sitkUInt8,
) -> sitk.Image:
    """Create (2D/3D) image with specified shape and spacing"""

    image = sitk.Image(*shape, pixel_type)
    if spacing:
        if len(shape) != len(spacing):
            raise ValueError("shape and spacing must have same dimension")
        image.SetSpacing(spacing)
    image[:] = value
    return image


def extract_slices(image: sitk.Image, axis: int = 2) -> List[sitk.Image]:
    """Get 2D image slices from 3D image

    Args:
        img (Image3): 3d image
        axis (int, optional): Axis perpendicular to slices. Defaults to 2, i.e. XY slices

    Returns:
        List[Image2]: [description]
    """
    slices = []

    size = list(image.GetSize())
    size[axis] = 1
    index = [0, 0, 0]

    for k in range(size[axis]):
        index[axis] = k
        slices.append(sitk.Extract(image, size, index, _COLLAPSE_STRATEGY_SUBMATRIX))
    return slices


def resample(
    image: sitk.Image, target_spacing: Sequence[float], nearest: bool = False
) -> sitk.Image:
    """resample (2D/3D) image to a target spacing"""

    size = list(image.GetSize())
    spacing = list(image.GetSpacing())
    for d in range(image.GetDimension()):
        size[d] = math.ceil(size[d] * spacing[d] / target_spacing[d])
        spacing[d] = target_spacing[d]

    filter = sitk.ResampleImageFilter()
    filter.SetOutputDirection(image.GetDirection())
    filter.SetOutputSpacing(spacing)
    filter.SetOutputOrigin(image.GetOrigin())
    filter.SetSize(size)
    filter.SetOutputPixelType(image.GetPixelID())
    filter.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    filter.SetDefaultPixelValue(0)

    filter.SetTransform(sitk.Transform())  # identity
    resampled: sitk.Image = filter.Execute(image)
    return resampled


def apply_transform(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    transform: sitk.Transform,
    nearest: bool,
) -> sitk.Image:
    """Resample and transform an image

    Description:
        The moving image is resampled to the reference grid of the
        fixed image. The transform maps a position in the fixed image
        to the moving image.
    """
    filter = sitk.ResampleImageFilter()
    filter.SetOutputDirection(fixed_image.GetDirection())
    filter.SetOutputSpacing(fixed_image.GetSpacing())
    filter.SetOutputOrigin(fixed_image.GetOrigin())
    filter.SetSize(fixed_image.GetSize())
    filter.SetOutputPixelType(moving_image.GetPixelID())
    filter.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    filter.SetDefaultPixelValue(0)

    filter.SetTransform(transform)
    resampled: sitk.Image = filter.Execute(moving_image)
    return resampled


def resample_to_ref(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    nearest: bool,
) -> sitk.Image:
    """resample (2D/3D) image to a reference grid

    Args:
        img: input image
        ref: reference image

    Returns:
        resampled image
    """
    return apply_transform(
        moving_image=moving_image,
        fixed_image=fixed_image,
        transform=sitk.Transform(),  # identity
        nearest=nearest,
    )


def pad(image: sitk.Image, target_size: Sequence[int], value: float = 0) -> sitk.Image:
    """Pad (2D/3D) image to the target size"""
    size = image.GetSize()
    delta = [max(s, t) - t for s, t in zip(size, target_size)]

    if any(delta):
        pad_low = [(d + 1) // 2 for d in delta]
        pad_hi = [delta[i] - p for i, p in enumerate(pad_low)]

        image = sitk.ConstantPad(image, pad_low, pad_hi, value)
    return image


def crop_center(image: sitk.Image, target_size: Sequence[int]) -> sitk.Image:
    """Crop (2D/3D) image to the target size (centered)"""
    size = image.GetSize()
    delta = [max(s, t) - t for s, t in zip(size, target_size)]

    if any(delta):
        crop_low = [(d + 1) // 2 for d in delta]
        crop_hi = [delta[i] - p for i, p in enumerate(crop_low)]

        image = sitk.Crop(image, crop_low, crop_hi)
    return image


def crop(
    img: sitk.Image, target_offset: Sequence[int], target_size: Sequence[int]
) -> sitk.Image:
    """Crop (2D/3D) image to the target size/offset"""
    cropped: sitk.Image = sitk.Extract(
        img, target_size, target_offset, _COLLAPSE_STRATEGY_SUBMATRIX
    )
    return cropped
