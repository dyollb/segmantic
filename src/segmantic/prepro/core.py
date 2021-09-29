import math
import numpy as np
import itk
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable

# frequently used types
from itk.itkImagePython import itkImageBase2 as Image2
from itk.itkImagePython import itkImageBase3 as Image3
from itk.support.types import ImageLike as AnyImage

itkImage = Union[Image2, Image3]
ImageOrArray = Union[Image2, Image3, np.ndarray]


def identity(x: AnyImage) -> AnyImage:
    return x


def as_image(x: AnyImage) -> AnyImage:
    if isinstance(x, np.ndarray):
        return itk.image_view_from_array(x)
    return x


def as_array(x: AnyImage) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return itk.array_from_image(x)


def extract_slices(img: Image3, axis: int = 2) -> List[Image2]:
    """Get 2D image slices from 3D image

    Args:
        img (Image3): 3d image
        axis (int, optional): Axis perpendicular to slices. Defaults to 2, i.e. XY slices

    Returns:
        List[Image2]: [description]
    """
    slices = []
    size = itk.size(img)

    region = itk.region(img)
    region.SetSize(axis, 1)
    _SUBMATRIX = 2

    for k in range(size[axis]):
        region.SetIndex(axis, k)
        slices.append(
            itk.extract_image_filter(
                img, extraction_region=region, direction_collapse_to_strategy=_SUBMATRIX
            )
        )
    return slices


def scale_to_range(img: AnyImage, vmin: float = 0.0, vmax: float = 255.0) -> AnyImage:
    """Scale numpy itk.Image to fit in range [vmin,vmax]"""
    x_view = as_array(img)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view += vmin - x_min
    np.multiply(x_view, vmax / (x_max - x_min), out=x_view, casting="unsafe")
    np.clip(x_view, a_min=vmin, a_max=vmax, out=x_view)
    return img


def resample(img: itkImage, target_spacing: Optional[Sequence] = None) -> itkImage:
    """resample N-D itk. Image to a target spacing

    Args:
        img (itkImage): input image
        target_spacing (Optional[Sequence]): target spacing (defaults to 0.85)

    Returns:
        itkImage: resampled image
    """
    dim = img.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    transform = itk.IdentityTransform[itk.D, dim].New()

    if not target_spacing:
        target_spacing = [0.85] * dim

    size = itk.size(img)
    spacing = itk.spacing(img)
    for d in range(dim):
        size[d] = math.ceil(size[d] * spacing[d] / target_spacing[d])
        spacing[d] = target_spacing[d]

    # resample to target resolution
    resampled = itk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=size,
        output_spacing=spacing,
        output_origin=itk.origin(img),
        output_direction=img.GetDirection(),
    )
    return resampled


def pad(img: AnyImage, target_size: tuple = (256, 256), value: float = 0) -> AnyImage:
    """Pad (2D) image to the target size"""
    size = itk.size(img)
    delta = [int(t - min(s, t)) for s, t in zip(size, target_size)]

    if any(delta):
        pad_lo = [(d + 1) // 2 for d in delta]
        pad_hi = [delta[i] - p for i, p in enumerate(pad_lo)]
        img = itk.constant_pad_image_filter(
            img,
            pad_lower_bound=pad_lo,
            pad_upper_bound=pad_hi,
            constant=value,
        )
    return img


def crop(img: AnyImage, target_size: tuple = (256, 256)) -> AnyImage:
    """Crop (2D) image to the target size (centered)"""
    size = itk.size(img)
    delta = [int(max(s, t) - t) for s, t in zip(size, target_size)]

    if any(delta):
        crop_low = [(d + 1) // 2 for d in delta]
        crop_hi = [delta[i] - p for i, p in enumerate(crop_low)]

        _SUBMATRIX = 2
        img = itk.crop_image_filter(
            img,
            lower_boundary_crop_size=crop_low,
            upper_boundary_crop_size=crop_hi,
            direction_collapse_to_strategy=_SUBMATRIX,
        )
    return img


def get_files(
    dir: Path, predicate: Callable[[str], bool] = lambda f: f.endswith(".nii.gz")
) -> List[Path]:
    """Collect list of file names filtered by 'predicate'"""
    return [f for f in Path(dir).glob("*.*") if predicate(f"{f}")]
