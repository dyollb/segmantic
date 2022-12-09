import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from segmantic.transforms.distance import DistanceTransform, DistanceTransformd


def test_DistanceTransform_2d():
    mask = torch.zeros(8, 9, dtype=torch.bool)
    mask[2:5, 3:6] = 1

    distance_transform = DistanceTransform()
    df = distance_transform(mask)
    assert isinstance(df, torch.Tensor)
    assert df.shape == (
        2,
        8,
        9,
    )


def test_DistanceTransform_3d():
    mask = torch.zeros(8, 9, 7, dtype=torch.bool)
    mask[2:5, 3:6, 3:5] = 1

    distance_transform = DistanceTransform()
    df = distance_transform(mask)
    assert isinstance(df, torch.Tensor)
    assert df.shape == (
        2,
        8,
        9,
        7,
    )


def test_DistanceTransformd():
    mask = torch.zeros(8, 9, dtype=torch.bool)
    mask[2:5, 3:6] = 1

    distance_transform = DistanceTransformd(keys="label", output_keys="dist")
    df = distance_transform({"label": mask})
    assert isinstance(df, dict)
    assert "dist" in df
    assert isinstance(df["dist"], torch.Tensor)


def test_DistanceTransform_MultiClass():
    mask = np.zeros((6, 7), dtype=int)
    mask[2, 3] = 1
    mask[4, 5] = 2

    spacing = (1.2, 1.7)

    distance_transform = DistanceTransform(num_classes=3, spacing=spacing)
    df = distance_transform(mask)
    assert df.shape == (
        3,
        6,
        7,
    )
    ref1 = distance_transform_edt(~(mask == 1), sampling=spacing)
    ref2 = distance_transform_edt(~(mask == 2), sampling=spacing)
    np.testing.assert_almost_equal(df[1, ...], ref1, decimal=6)
    np.testing.assert_almost_equal(df[2, ...], ref2, decimal=6)
