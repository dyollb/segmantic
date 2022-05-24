import numpy as np
import pytest
import torch

from segmantic.seg.normalize import Interpolate1d


def test_interpolate1d():
    x = np.array([0, 4, 6, 10])
    y = np.array([1, -1, 3, 5])

    interp1d = Interpolate1d(x, y)
    yi = interp1d(torch.tensor([0, 2, 4, 8, 10]))
    print(yi)
    assert yi[0] == pytest.approx(1.0, 1e-3)
    assert yi[1] == pytest.approx(0.0, 1e-3)
    assert yi[2] == pytest.approx(-1.0, 1e-3)
    assert yi[3] == pytest.approx(4.0, 1e-3)
    assert yi[4] == pytest.approx(5.0, 1e-3)

    yi = interp1d(torch.tensor([-1, 11]))
    print(yi)
    assert yi[0] == pytest.approx(1.5, 1e-3)
    assert yi[1] == pytest.approx(5.5, 1e-3)

    yi = interp1d(torch.tensor([[-2, 11], [1, 3], [8, 10]]))
    print(yi)
    assert yi.shape == (3, 2)
    assert yi[0, 0] == pytest.approx(2.0, 1e-3)
    assert yi[0, 1] == pytest.approx(5.5, 1e-3)
    assert yi[1, 0] == pytest.approx(0.5, 1e-3)
    assert yi[1, 1] == pytest.approx(-0.5, 1e-3)
    assert yi[2, 0] == pytest.approx(4.0, 1e-3)
    assert yi[2, 1] == pytest.approx(5.0, 1e-3)

    # assert yi.device.type ==
    if torch.cuda.is_available():
        x = torch.tensor([-2, 11], device=torch.device("cuda"))
        yi = interp1d(x)
        assert x.device == yi.device
