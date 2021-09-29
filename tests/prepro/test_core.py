import itk
import numpy as np

import pytest
from segmantic.prepro import core


@pytest.fixture
def labelfield() -> core.Image3:
    """3D labelfield, where each slice has a uniform label = slice number"""
    region = itk.ImageRegion[3]()
    region.SetSize((5, 5, 5))
    region.SetIndex((0, 0, 0))

    image = itk.Image[itk.UC, 3].New()
    image.SetRegions(region)
    image.SetSpacing((0.5, 0.6, 0.7))
    image.Allocate()

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


def test_resample():
    region = itk.ImageRegion[2]()
    region.SetSize((3, 3))

    image = itk.Image[itk.F, 2].New()
    image.SetRegions(region)
    image.SetSpacing((2.0, 2.0))
    image.Allocate()

    image[:] = 1.0
    image[1, 1] = 0.0

    # double the resolution from (2.0, 2.0) to (1.0, 1.0)
    res = core.resample(image, target_spacing=(1.0, 1.0))

    assert list(res.shape) == [2 * s for s in image.shape]
