import math
import os
import numpy as np
import itk
from typing import Optional, Union


image_3d = itk.itkImagePython.itkImageBase3
image_2d = itk.itkImagePython.itkImageBase2
any_image = Union[image_2d, image_3d, np.ndarray]


def identity(x: any_image):
    return x


def as_image(x: any_image):
    if isinstance(x, np.ndarray):
        return itk.image_view_from_array(x)
    return x


def as_array(x: any_image):
    if isinstance(x, np.ndarray):
        return x
    return itk.array_from_image(x)


def extract_slices(img: image_3d, axis: int = 2):
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


def scale_to_range(img: any_image, vmin: float = 0.0, vmax: float = 255.0):
    """Scale numpy itk.Image to fit in range [vmin,vmax]"""
    x_view = as_array(img)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view += vmin - x_min
    np.multiply(x_view, vmax / (x_max - x_min), out=x_view, casting="unsafe")
    np.clip(x_view, a_min=vmin, a_max=vmax, out=x_view)
    return img


def resample(img: Union[image_2d, image_3d], target_spacing=Optional[tuple]):
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


def pad_slice(img: image_2d, target_size: tuple = (256, 256), value: float = 0):
    ''' Pad (2D) image to the target size '''
    size = itk.size(img)
    delta = (t - min(s, t) for s, t in zip(size, target_size))

    if any(delta):
        pad_lo = ((d + 1) // 2 for d in delta)
        pad_hi = (delta[i] - p for i, p in enumerate(pad_lo))
        img = itk.constant_pad_image_filter(
            img,
            pad_lower_bound=pad_lo,
            pad_upper_bound=pad_hi,
            constant=value,
        )
    return img


def get_files(dir: str, cond=lambda x: True, ext: str = ".nii.gz"):
    return [
        os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext) and cond(f)
    ]

