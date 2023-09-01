import numpy as np
import pytest
import torch

from segmantic.seg.nyul_normalize import NyulNormalize


def test_interp1d():
    xp = torch.tensor([0.0, 4, 6, 10])
    yp = torch.tensor([1.0, -1, 3, 5])

    transform = NyulNormalize(
        quantiles=np.array([0.1, 0.5, 0.9]), standard_scale=np.array([0, 0.5, 1.0])
    )

    yi = transform.interp1d(torch.tensor([0.0, 2, 4, 8, 10]), xp, yp)
    assert yi.shape == (5,)
    assert yi == pytest.approx(torch.tensor([1.0, 0.0, -1.0, 4.0, 5.0]), 1e-3)

    yi = transform.interp1d(torch.tensor([-1.0, 11]), xp, yp)
    assert yi.shape == (2,)
    assert yi == pytest.approx(torch.tensor([1.5, 5.5]), 1e-3)

    yi = transform.interp1d(torch.tensor([[-2.0, 11], [1, 3], [8, 10]]), xp, yp)
    assert yi.shape == (3, 2)
    assert yi == pytest.approx(
        torch.tensor([[2.0, 5.5], [0.5, -0.5], [4.0, 5.0]]), 1e-3
    )

    # assert yi.device.type ==
    if torch.cuda.is_available():
        x = torch.tensor([-2.0, 11], device=torch.device("cuda"))
        xp, yp = xp.to(x.device), yp.to(x.device)
        yi = transform.interp1d(x, xp, yp)
        assert x.device == yi.device
