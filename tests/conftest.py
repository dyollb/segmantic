import pytest
import SimpleITK as sitk

from segmantic.image.processing import make_image


@pytest.fixture
def labelfield() -> sitk.Image:
    """3D labelfield, where each XY slice has a uniform label = slice number"""
    image = make_image(shape=(5, 5, 5), spacing=(0.5, 0.6, 0.7))
    for i in range(5):
        image[i, :, :] = i
    return image
