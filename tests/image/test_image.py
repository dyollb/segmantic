import numpy as np
import SimpleITK as sitk

from segmantic.image import processing


def test_extract_slices(labelfield: sitk.Image) -> None:
    slices_xy = processing.extract_slices(labelfield, axis=2)

    assert slices_xy[0].GetSpacing()[0] == labelfield.GetSpacing()[0]
    assert slices_xy[0].GetSpacing()[1] == labelfield.GetSpacing()[1]

    for k, slice in enumerate(slices_xy):
        slice_view = sitk.GetArrayViewFromImage(slice)
        assert np.all(slice_view == k)


def test_pad_crop_center(labelfield: sitk.Image) -> None:
    padded = processing.pad(labelfield, target_size=(9, 9, 9))
    cropped = processing.crop_center(padded, target_size=(5, 5, 5))

    assert labelfield.GetSpacing() == cropped.GetSpacing()
    assert labelfield.GetOrigin() == cropped.GetOrigin()
    assert np.all(
        sitk.GetArrayViewFromImage(cropped) == sitk.GetArrayViewFromImage(labelfield)
    )

    slice = processing.crop_center(labelfield, target_size=(5, 5, 1))
    size = slice.GetSize()
    assert size[2] == 1


def test_resample(labelfield: sitk.Image) -> None:
    # half the spacing
    spacing = [s / 2.0 for s in labelfield.GetSpacing()]
    res = processing.resample(labelfield, target_spacing=spacing)

    # expecting double the resolution
    assert list(res.GetSize()) == [2 * s for s in labelfield.GetSize()]
