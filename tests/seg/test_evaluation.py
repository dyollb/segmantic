import itk
import numpy as np
from typing import Sequence
from segmantic.prepro.core import Image3

import pytest
from segmantic.seg import evaluation


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


def test_confusion_matrix():
    labelfield = make_image(shape=(10, 10), spacing=(1.0, 1.0), value=0)
    labelfield[2:3, 2:4] = 1
    labelfield[3:5, 3:4] = 2

    view = itk.array_view_from_image(labelfield).flatten()
    num_classes = int(np.max(view) + 1)
    assert num_classes == 3

    count = np.array(np.bincount(view))

    cm = evaluation.compute_confusion(num_classes, view, view)
    assert np.all(np.diagonal(cm) == count)
    assert np.all(np.diagonal(cm, offset=1) == 0)
    assert np.all(np.diagonal(cm, offset=-1) == 0)


def test_hausdorff():
    labelfield = make_image(shape=(10, 10), spacing=(1.0, 1.0))
    labelfield[3:6, 3:6] = 1

    r11 = evaluation.hausdorff_distance(labelfield, labelfield)
    assert r11["mean"] == 0.0
    assert all(v == 0.0 for v in r11.values())

    labelfield2 = make_image(shape=(10, 10), spacing=(1.0, 1.0))
    labelfield2[1:8, 2:7] = 1

    r12 = evaluation.hausdorff_distance(labelfield, labelfield2)
    assert r12["max"] >= 2.0
    assert all(v > 0.0 for v in r12.values())
