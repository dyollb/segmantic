from typing import Dict, Optional, Sequence, Union

import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.networks.utils import one_hot
from monai.transforms.post.array import Ensemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import Transform
from monai.utils import TransformBackends


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

    def __init__(self, label_model_dict: Dict[int, int]) -> None:
        self.label_model_dict = label_model_dict

    def __call__(
        self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]
    ) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)

        if img_.ndimension() > 1 and img_.shape[1] > 1:
            # if multi-channel (One-Hot) images are passed
            img_ = torch.argmax(img_, dim=1, keepdim=True)
            has_ch_dim = True
        else:
            # combining the tissues from the best performing models
            has_ch_dim = False

        final_img = torch.empty(img_.size()[1:])
        for tissue_id, model_id in self.label_model_dict.items():
            temp_best_tissue = img_[model_id, :, :, :, :]
            final_img[temp_best_tissue == tissue_id] = tissue_id

        if has_ch_dim:
            num_classes = max(self.label_model_dict.keys()) + 1
            out_pt = one_hot(final_img, num_classes, dim=0)
        else:
            out_pt = final_img

        return self.post_convert(out_pt, img)


class SelectBestEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SelectBestEnsemble`.
    """

    backend = SelectBestEnsemble.backend

    def __init__(
        self,
        label_model_dict: Dict[int, int],
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
