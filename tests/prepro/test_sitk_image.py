import numpy as np
import SimpleITK as sitk

from segmantic.prepro import sitk_image
from segmantic.prepro.sitk_image import make_image


def test_image() -> sitk.Image:
    data = np.zeros((6, 4, 4))
    for i in range(4):
        data[..., i] = i
    img = sitk.GetImageFromArray(data)
    img.SetSpacing((1.2, 1.3, 2.4))
    return img


def test_extract_slices() -> None:
    labelfield = test_image()
    slices_xy = sitk_image.extract_slices(labelfield, axis=2)

    assert slices_xy[0].GetSpacing()[0] == labelfield.GetSpacing()[0]
    assert slices_xy[0].GetSpacing()[1] == labelfield.GetSpacing()[1]

    for k, slice in enumerate(slices_xy):
        slice_view = sitk.GetArrayViewFromImage(slice)
        assert np.all(slice_view == k)


def test_pad_crop_center(labelfield: sitk.Image) -> None:
    labelfield = test_image()
    padded = sitk_image.pad(labelfield, target_size=(9, 9, 9))
    cropped = sitk_image.crop_center(padded, target_size=(5, 5, 5))

    assert labelfield.GetSpacing() == cropped.GetSpacing()
    assert labelfield.GetOrigin() == cropped.GetOrigin()
    assert np.all(
        sitk.GetArrayViewFromImage(cropped) == sitk.GetArrayViewFromImage(labelfield)
    )

    slice = sitk_image.crop_center(labelfield, target_size=(5, 5, 1))
    size = slice.GetSize()
    assert size[2] == 1


def test_resample() -> None:
    image = make_image(
        shape=(3, 3), spacing=(2.0, 2.0), value=1.0, pixel_type=sitk.sitkFloat32
    )
    image[1, 1] = 0.0

    # double the resolution from (2.0, 2.0) to (1.0, 1.0)
    res = sitk_image.resample(image, target_spacing=(1.0, 1.0))

    assert list(res.shape) == [2 * s for s in image.shape]
