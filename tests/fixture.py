import itk
from itk.support.types import itkCType
from typing import Sequence, Union

from segmantic.prepro.core import Image3


def make_image(
    shape: Sequence[int],
    spacing: Sequence[float],
    value: Union[int, float] = 0,
    pixel_type: itkCType = itk.UC,
) -> Image3:
    """Create image with specified shape and spacing"""
    assert len(shape) == len(spacing)
    dim = len(shape)

    region = itk.ImageRegion[dim]()
    region.SetSize(shape)
    region.SetIndex(tuple([0] * dim))

    image = itk.Image[pixel_type, dim].New()
    image.SetRegions(region)
    image.SetSpacing(spacing)
    image.Allocate()

    image[:] = value
    return image
