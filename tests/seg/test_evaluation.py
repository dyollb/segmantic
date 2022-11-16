import numpy as np
import SimpleITK as sitk

from segmantic.image.processing import make_image
from segmantic.seg import evaluation


def test_confusion_matrix():
    labelfield = make_image(shape=(10, 10), spacing=(1.0, 1.0), value=0)
    labelfield[2:3, 2:4] = 1
    labelfield[3:5, 3:4] = 2

    view = sitk.GetArrayViewFromImage(labelfield).flatten()
    num_classes = int(np.max(view) + 1)
    assert num_classes == 3

    count = np.array(np.bincount(view))

    cm = evaluation.confusion_matrix(num_classes, view, view)
    assert np.all(np.diagonal(cm) == count)
    assert np.all(np.diagonal(cm, offset=1) == 0)
    assert np.all(np.diagonal(cm, offset=-1) == 0)


def test_hausdorff():
    labelfield = make_image(shape=(10, 10), spacing=(1.0, 1.0))
    labelfield[3:6, 3:6] = 1

    r11 = evaluation.hausdorff_surface_distance(labelfield, labelfield)
    assert r11["mean"] == 0.0
    assert all(v == 0.0 for v in r11.values())

    labelfield2 = make_image(shape=(10, 10), spacing=(1.0, 1.0))
    labelfield2[1:8, 2:7] = 1

    r12 = evaluation.hausdorff_surface_distance(labelfield, labelfield2)
    assert r12["max"] >= 2.0
    assert all(v > 0.0 for v in r12.values())
