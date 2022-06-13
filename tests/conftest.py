import itk
import pytest

from segmantic.prepro.core import Image3, make_image


@pytest.fixture
def labelfield() -> Image3:
    """3D labelfield, where each XY slice has a uniform label = slice number"""
    image = make_image(shape=(5, 5, 5), spacing=(0.5, 0.6, 0.7))
    view = itk.array_view_from_image(image)
    for i in range(5):
        # note: itk exposes the x-fastest array to numpy in c-order, i.e. view[z,y,x]
        view[i, :, :] = i
    return image
