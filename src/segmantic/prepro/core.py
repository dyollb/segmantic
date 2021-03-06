import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TypeVar, Union

import itk
import numpy as np

# frequently used types
from itk.itkImagePython import itkImageBase2 as Image2
from itk.itkImagePython import itkImageBase3 as Image3
from itk.support.types import itkCType

#: ImageAnyd: Union of Image2 and Image3 used for typing
ImageAnyd = Union[Image2, Image3]

# Generic type which can represent either an Image2 or an Image3
# Unlike Union can create a dependence between parameter(s) / return(s)
ImageNd = TypeVar("ImageNd", bound=ImageAnyd)

#: NdarrayOrImage: Union of numpy.ndarray and Image2/Image3 to be used for typing
NdarrayOrImage = Union[Image2, Image3, np.ndarray]

# Generic type which can represent either a numpy.ndarray or an Image2/Image3
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayImage = TypeVar("NdarrayImage", bound=NdarrayOrImage)

_COLLAPSE_STRATEGY_SUBMATRIX = 2


def identity(x: NdarrayImage) -> NdarrayImage:
    return x


def as_image(x: NdarrayOrImage) -> ImageAnyd:
    if isinstance(x, np.ndarray):
        return itk.image_view_from_array(x)
    return x


def as_array(x: NdarrayOrImage) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return itk.array_from_image(x)  # type: ignore


def array_view_reverse_ordering(x: np.ndarray) -> np.ndarray:
    return x.transpose(np.flip(np.arange(len(x.shape))))


def imread(filename: Path) -> ImageAnyd:
    """Wrapper around itk.imread to avoid having to convert Path to str"""
    return itk.imread(f"{filename}")


def imwrite(image: ImageAnyd, filename: Path, compression: bool = False) -> None:
    """Wrapper around itk.imwrite to avoid having to convert Path to str"""
    itk.imwrite(image, f"{filename}", compression=compression)


def pixeltype(image: ImageAnyd) -> itkCType:
    """Get pixel type"""
    return itk.template(image)[1][0]


def make_image(
    shape: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    value: Union[int, float] = 0,
    pixel_type: itkCType = itk.UC,
) -> ImageAnyd:
    """Create (2D/3D) image with specified shape and spacing"""
    dim = len(shape)

    region = itk.ImageRegion[dim]()
    region.SetSize(shape)
    region.SetIndex(tuple([0] * dim))

    image = itk.Image[pixel_type, dim].New()
    image.SetRegions(region)
    if spacing:
        if len(shape) != len(spacing):
            raise ValueError("shape and spacing must have same dimension")
        image.SetSpacing(spacing)
    image.Allocate()

    image[:] = value
    return image


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

    for k in range(size[axis]):
        region.SetIndex(axis, k)
        slices.append(
            itk.extract_image_filter(
                img,
                extraction_region=region,
                direction_collapse_to_strategy=_COLLAPSE_STRATEGY_SUBMATRIX,
            )
        )
    return slices


def scale_to_range(
    img: NdarrayImage, vmin: float = 0.0, vmax: float = 255.0
) -> NdarrayImage:
    """Scale numpy itk.Image to fit in range [vmin,vmax]"""
    x_view = as_array(img)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view += vmin - x_min
    np.multiply(x_view, vmax / (x_max - x_min), out=x_view, casting="unsafe")
    np.clip(x_view, a_min=vmin, a_max=vmax, out=x_view)
    return img


def resample(img: ImageNd, target_spacing: Optional[Sequence] = None) -> ImageNd:
    """resample (2D/3D) image to a target spacing

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
    resampled: ImageNd = itk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=size,
        output_spacing=spacing,
        output_origin=itk.origin(img),
        output_direction=img.GetDirection(),
    )
    return resampled


def resample_to_ref(img: ImageNd, ref: ImageNd) -> ImageNd:
    """resample (2D/3D) image to a reference grid

    Args:
        img: input image
        ref: reference image

    Returns:
        itkImage: resampled image
    """
    dim = img.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    transform = itk.IdentityTransform[itk.D, dim].New()

    # resample to target resolution
    resampled: ImageNd = itk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=itk.size(ref),
        output_spacing=itk.spacing(ref),
        output_origin=itk.origin(ref),
        output_direction=ref.GetDirection(),
    )
    return resampled


def pad(
    img: ImageNd, target_size: Sequence[int] = (256, 256), value: float = 0
) -> ImageNd:
    """Pad (2D/3D) image to the target size"""
    size = itk.size(img)
    delta = [t - min(s, t) for s, t in zip(size, target_size)]

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


def crop_center(img: ImageAnyd, target_size: Sequence[int] = (256, 256)) -> ImageAnyd:
    """Crop (2D/3D) image to the target size (centered)"""
    size = itk.size(img)
    delta = [max(s, t) - t for s, t in zip(size, target_size)]

    if any(delta):
        crop_low = [(d + 1) // 2 for d in delta]
        crop_hi = [delta[i] - p for i, p in enumerate(crop_low)]

        img = itk.crop_image_filter(
            img,
            lower_boundary_crop_size=crop_low,
            upper_boundary_crop_size=crop_hi,
            direction_collapse_to_strategy=_COLLAPSE_STRATEGY_SUBMATRIX,
        )
    return img


def crop(
    img: ImageAnyd,
    target_offset: Sequence[int],
    target_size: Sequence[int] = (256, 256),
) -> ImageAnyd:
    """Crop (2D/3D) image to the target size/offset"""
    region = itk.region(img)
    region.SetIndex(target_offset)
    region.SetSize(target_size)

    return itk.extract_image_filter(
        img,
        extraction_region=region,
        direction_collapse_to_strategy=_COLLAPSE_STRATEGY_SUBMATRIX,
    )


def get_files(
    dir: Path, predicate: Callable[[str], bool] = lambda f: f.endswith(".nii.gz")
) -> List[Path]:
    """Collect list of file names filtered by 'predicate'"""
    return [f for f in Path(dir).glob("*.*") if predicate(f"{f}")]
