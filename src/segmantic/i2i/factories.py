import torch
import torch.nn as nn

if torch.torch_version.TorchVersion >= (1, 10, 0):  # type: ignore
    from torch.nn.modules.padding import ReflectionPad3d
else:
    from .backports import ReflectionPad3d  # type: ignore

from typing import Type, Union

from monai.networks.layers.factories import Act, Conv, Norm, Pad  # noqa: F401


@Pad.factory_function("reflectionpad")
def reflection_pad_factory(
    dim: int,
) -> Type[Union[nn.ReflectionPad1d, nn.ReflectionPad2d, ReflectionPad3d]]:
    types = (nn.ReflectionPad1d, nn.ReflectionPad2d, ReflectionPad3d)
    return types[dim - 1]
