import itk
import numpy as np

import pytest
from segmantic.prepro import core
from tests.fixture import make_image


@pytest.fixture
def labelfield() -> core.Image3:
    """3D labelfield, where each slice has a uniform label = slice number"""
    image = make_image(shape=(5, 5, 5), spacing=(0.5, 0.6, 0.7))
    view = itk.array_view_from_image(image)
    for i in range(5):
        # note: itk exposes the x-fastest array to numpy in c-order, i.e. view[z,y,x]
        view[i, :, :] = i
    return image


def test_extract_slices(labelfield: core.Image3):

    slices_xy = core.extract_slices(labelfield, axis=2)

    assert slices_xy[0].GetSpacing()[0] == labelfield.GetSpacing()[0]
    assert slices_xy[0].GetSpacing()[1] == labelfield.GetSpacing()[1]

    for k, slice in enumerate(slices_xy):
        print(type(slice))
        slice_view = itk.array_view_from_image(slice)
        assert np.all(slice_view == k)


def test_pad_crop(labelfield: core.Image3):
    padded = core.pad(labelfield, target_size=(9, 9, 9))
    cropped = core.crop(padded, target_size=(5, 5, 5))

    assert labelfield.GetSpacing() == cropped.GetSpacing()
    assert labelfield.GetOrigin() == cropped.GetOrigin()
    assert np.all(core.as_array(cropped) == core.as_array(labelfield))

    slice = core.crop(labelfield, target_size=(5, 5, 1))
    size = itk.size(slice)
    assert size[2] == 1


def test_resample():
    image = make_image(shape=(3, 3), spacing=(2.0, 2.0), value=1.0, pixel_type=itk.F)
    image[1, 1] = 0.0

    # double the resolution from (2.0, 2.0) to (1.0, 1.0)
    res = core.resample(image, target_spacing=(1.0, 1.0))

    assert list(res.shape) == [2 * s for s in image.shape]
