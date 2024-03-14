from collections.abc import Hashable, Mapping, Sequence
from typing import Optional, Union

import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.data.meta_obj import get_track_meta
from monai.networks.utils import one_hot
from monai.transforms.post.array import Ensemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import MapTransform, Transform
from monai.utils import TransformBackends, convert_to_dst_type, convert_to_tensor


class SelectBestEnsemble(Ensemble, Transform):
    """
    Execute select best ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E[, C, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.

    Note:
        This select best transform expects the input data is discrete single channel values.
        It selects the tissue of the model which performed best in a generalization analysis.
        The mapping is saved in the label_model_dict.
        The output data has the same shape as every item of the input data.

    Args:
        label_model_dict: dictionary containing the best models index for each tissue and
        the tissue labels.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, label_model_dict: dict[int, int]) -> None:
        self.label_model_dict = label_model_dict

    def __call__(
        self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]
    ) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)

        has_ch_dim = False
        if img_.ndimension() > 1 and img_.shape[1] > 1:
            # convert multi-channel (One-Hot) images to argmax
            img_ = torch.argmax(img_, dim=1, keepdim=True)
            has_ch_dim = True

        # combining the tissues from the best performing models
        out_pt = torch.empty(img_.size()[1:])
        for tissue_id, model_id in self.label_model_dict.items():
            temp_best_tissue = img_[model_id, ...]
            out_pt[temp_best_tissue == tissue_id] = tissue_id

        if has_ch_dim:
            # convert back to multi-channel (One-Hot)
            num_classes = max(self.label_model_dict.keys()) + 1
            out_pt = one_hot(out_pt, num_classes, dim=0)

        return self.post_convert(out_pt, img)


class SelectBestEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SelectBestEnsemble`.
    """

    backend = SelectBestEnsemble.backend

    def __init__(
        self,
        label_model_dict: dict[int, int],
        keys: KeysCollection,
        output_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.

        """
        ensemble = SelectBestEnsemble(
            label_model_dict=label_model_dict,
        )
        super().__init__(keys, ensemble, output_key)


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
