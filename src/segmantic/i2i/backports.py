from typing import Tuple

import torch
from torch.nn.modules.padding import _ntuple, _ReflectionPadNd, _size_6_t


class ReflectionPad3d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    NOTE: ReflectionPad3d has been added in torch, but is not yet released in v1.10
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padding = self.padding
        in_shape: Tuple[int, ...] = input.shape
        paddable_shape = in_shape[2:]

        # Get shape of padded tensor
        out_shape = in_shape[:2]
        for idx, size in enumerate(paddable_shape):
            assert padding[-(idx * 2 + 1)] < size
            assert padding[-(idx * 2 + 2)] < size
            out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)

        out = torch.zeros(
            out_shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

        # Put original array in padded array
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        out_w0 = max(padding[-6], 0)
        out_w1 = out_shape[4] - max(padding[-5], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        in_w0 = max(-padding[-6], 0)
        in_w1 = in_shape[4] - max(-padding[-5], 0)

        out[..., out_d0:out_d1, out_h0:out_h1, out_w0:out_w1] = input[
            ..., in_d0:in_d1, in_h0:in_h1, in_w0:in_w1
        ]

        # The following steps first pad the beginning of the tensor (left side),
        # and then pad the end of the tensor (right side).
        # Note: Corners will be written more than once when ndim > 1.

        # Copying is only performed where padding values are > 0.

        # Pad first dimension (depth)
        if padding[-2] > 0:
            o0 = 0
            o1 = padding[-2]
            ids = [i for i in range(o1 + padding[-2], o1, -1)]
            out[:, :, o0:o1] = torch.index_select(out, 2, torch.LongTensor(ids))
        if padding[-1] > 0:
            o0 = out_shape[2] - padding[-1]
            o1 = out_shape[2]
            ids = [i - 2 for i in range(o0, o0 - padding[-1], -1)]
            out[:, :, o0:o1] = torch.index_select(out, 2, torch.LongTensor(ids))

        # Pad second dimension (height)
        if len(padding) > 2:
            if padding[-4] > 0:
                o0 = 0
                o1 = padding[-4]
                ids = [i for i in range(o1 + padding[-4], o1, -1)]
                out[:, :, :, o0:o1] = torch.index_select(out, 3, torch.LongTensor(ids))
            if padding[-3] > 0:
                o0 = out_shape[3] - padding[-3]
                o1 = out_shape[3]
                ids = [i - 2 for i in range(o0, o0 - padding[-3], -1)]
                out[:, :, :, o0:o1] = torch.index_select(out, 3, torch.LongTensor(ids))

        # Pad third dimension (width)
        if len(padding) > 4:
            if padding[-6] > 0:
                o0 = 0
                o1 = padding[-6]
                ids = [i for i in range(o1 + padding[-6], o1, -1)]
                out[:, :, :, :, o0:o1] = torch.index_select(
                    out, 4, torch.LongTensor(ids)
                )
            if padding[-5] > 0:
                o0 = out_shape[4] - padding[-5]
                o1 = out_shape[4]
                ids = [i - 2 for i in range(o0, o0 - padding[-5], -1)]
                out[:, :, :, :, o0:o1] = torch.index_select(
                    out, 4, torch.LongTensor(ids)
                )

        return out
