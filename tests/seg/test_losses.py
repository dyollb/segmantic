from __future__ import annotations

import numpy as np
import pytest
import torch

from segmantic.seg.losses import AsymmetricUnifiedFocalLoss

TEST_CASES = [
    (  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {
            "y_pred": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
            "y_true": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
        },
        0.0,
    ),
    (  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {
            "y_pred": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
            "y_true": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
        },
        0.0,
    ),
]


@pytest.mark.parametrize("input_data,expected_val", TEST_CASES)
def test_result(input_data, expected_val):
    loss = AsymmetricUnifiedFocalLoss()
    result = loss(**input_data)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4
    )


def test_ill_shape():
    loss = AsymmetricUnifiedFocalLoss()
    with pytest.raises(ValueError):
        loss(torch.ones((2, 2, 2)), torch.ones((2, 2, 2, 2)))


def test_with_cuda():
    loss = AsymmetricUnifiedFocalLoss()
    i = torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]])
    j = torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]])
    if torch.cuda.is_available():
        i = i.cuda()
        j = j.cuda()
    output = loss(i, j)
    np.testing.assert_allclose(output.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_with_cuda()
