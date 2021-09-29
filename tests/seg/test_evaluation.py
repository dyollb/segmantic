import itk
import numpy as np
from segmantic.prepro.core import Image3

import pytest
from segmantic.seg import evaluation
from tests.prepro.test_core import labelfield


def test_confusion_matrix(labelfield: Image3):
    view = itk.array_view_from_image(labelfield).flatten()
    num_classes = int(np.max(view) + 1)

    count = np.array(np.bincount(view))

    cm = evaluation.compute_confusion(num_classes, view, view)
    assert np.all(np.diagonal(cm) == count)
    assert np.all(np.diagonal(cm, offset=1) == 0)
    assert np.all(np.diagonal(cm, offset=-1) == 0)
