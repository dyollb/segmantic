from collections.abc import Hashable, Mapping

import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform, Transform
from monai.utils import TransformBackends, convert_to_dst_type, convert_to_tensor


class MapLabels(Transform):
    """ """

    backend = [TransformBackends.TORCH]

    def __init__(self, mapping: dict[int, int]) -> None:
        self.lookup = torch.zeros((max(mapping.keys()) + 1,), dtype=torch.int64)
        for k in mapping:
            self.lookup[k] = mapping[k]

    def __call__(self, img: NdarrayTensor) -> NdarrayTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        mapping, *_ = convert_to_dst_type(self.lookup, dst=img, dtype=self.lookup.dtype)
        return mapping[img]  # type: ignore [return-value]


class MapLabelsd(MapTransform):
    """ """

    backend = MapLabels.backend

    def __init__(
        self,
        mapping: dict[int, int],
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.converter = MapLabels(mapping)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class SelectChannel(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0, channel: int = 0) -> None:
        self.channel_dim = channel_dim
        self.channel = channel

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`
        """
        if isinstance(img, torch.Tensor):
            output = img.index_select(
                index=torch.tensor(self.channel, dtype=torch.long, device=img.device),
                dim=self.channel_dim,
            )
        else:
            output = img.take(indices=self.channel, axis=self.channel_dim)

        if isinstance(img, MetaTensor) and not isinstance(output, MetaTensor):
            output = MetaTensor(output, meta=img.meta)

        return output


class SelectChanneld(MapTransform):

    backend = SelectChannel.backend

    def __init__(
        self,
        keys: KeysCollection,
        channel_dim: int = 0,
        channel: int = 0,
        new_key_postfix: str = "_sel",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.postfix = new_key_postfix
        self.op = SelectChannel(channel_dim, channel)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[str(key) + self.postfix] = self.op(d[key])
        return d
