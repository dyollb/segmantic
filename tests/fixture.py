import itk
from typing import Sequence

from segmantic.prepro.core import Image3


def make_image(
    shape: Sequence[int], spacing: Sequence[float], value: int = 0
) -> Image3:
    """Create image with specified shape and spacing"""
    assert len(shape) == len(spacing)
    dim = len(shape)

    region = itk.ImageRegion[dim]()
    region.SetSize(shape)
    region.SetIndex(tuple([0] * dim))

    image = itk.Image[itk.UC, dim].New()
    image.SetRegions(region)
    image.SetSpacing(spacing)
    image.Allocate()

    image[:] = value
    return image
