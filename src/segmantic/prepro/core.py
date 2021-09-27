import math
import os
import numpy as np
import itk
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, Callable


# frequently used types
from itk.itkImagePython import itkImageBase2 as Image2
from itk.itkImagePython import itkImageBase3 as Image3
from itk.support.types import ImageLike as AnyImage
ImageNd = Union[Image2, Image3]
PathLike = Union[str, Path]


def identity(x: AnyImage) -> AnyImage:
    return x


def as_image(x: AnyImage) -> ImageNd:
    if isinstance(x, np.ndarray):
        return itk.image_view_from_array(x)
    return x


def as_array(x: AnyImage) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return itk.array_from_image(x)


def extract_slices(img: Image3, axis: int = 2) -> List[Image2]:
    """Get 2D image slices from 3D image"""
    slices = []
    for k in range(img.shape[axis]):
        if axis == 0:
            slices.append(img[k, :, :])
        elif axis == 1:
            slices.append(img[:, k, :])
        else:
            slices.append(img[:, :, k])
    return slices


def scale_to_range(img: AnyImage, vmin: float = 0.0, vmax: float = 255.0) -> AnyImage:
    """Scale numpy itk.Image to fit in range [vmin,vmax]"""
    x_view = as_array(img)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view += vmin - x_min
    np.multiply(x_view, vmax / (x_max - x_min), out=x_view, casting="unsafe")
    np.clip(x_view, a_min=vmin, a_max=vmax, out=x_view)
    return img


def resample(img: ImageNd, target_spacing: Optional[Sequence] = None) -> ImageNd:
    """resample N-D itk.Image to a fixed spacing (default:0.85)"""
    dim = img.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    transform = itk.IdentityTransform[itk.D, dim].New()

    if not target_spacing:
        target_spacing = [0.85] * dim

    size = itk.size(img)
    spacing = itk.spacing(img)
    for d in range(dim):
        size[d] = math.ceil(size[d] * spacing[d] / 0.85)
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
        print(size)
        print(crop_low)
        print(crop_hi)
        img = itk.crop_image_filter(
            img,
            lower_boundary_crop_size=crop_low,
            upper_boundary_crop_size=crop_hi,
        )
    return img


def get_files(dir: str, predicate: Callable[[str], bool]=lambda f: f.endswith(".nii.gz")) -> list:
    """Collect list of file names filtered by 'predicate' """
    return [os.path.join(dir, f) for f in os.listdir(dir) if predicate(f)]
