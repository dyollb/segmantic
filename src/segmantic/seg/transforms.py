from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.post.array import Ensemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import Transform
from monai.utils import TransformBackends

from ..util.config import load


class SelectBestEnsemble(Ensemble, Transform):
    """
    Execute vote ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E[, C, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.

    Note:
        This vote transform expects the input data is discrete values. It can be multiple channels
        data in One-Hot format or single channel data. It will vote to select the most common data
        between items.
        The output data has the same shape as every item of the input data.

    Args:
        num_classes: if the input is single channel data instead of One-Hot, we can't get class number
            from channel, need to explicitly specify the number of classes to vote.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        candidate_per_tissue_path: Path,
        tissue_dict: Dict[str, int],
    ) -> None:

        self.tissue_dict = tissue_dict
        self.best_candidate_for_tissue = load(config_file=candidate_per_tissue_path)

    def __call__(
        self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]
    ) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)

        final_img = torch.empty(img_.size()[1:])
        if img_.size()[1] > 1:
            for tissue_name, tissue_lbl in self.tissue_dict.items():
                best_idx = self.best_candidate_for_tissue[tissue_name]
                final_img[tissue_lbl, :, :, :] = img_[best_idx, tissue_lbl, :, :, :]

            out_pt = torch.argmax(final_img, dim=0, keepdim=True)
        else:
            for tissue_name, model_nr in self.best_candidate_for_tissue.items():
                print(tissue_name, model_nr)
                tissue_lbl = self.tissue_dict[tissue_name]

                temp_best_tissue = img_[model_nr, :, :, :, :]
                final_img[temp_best_tissue == tissue_lbl] = tissue_lbl

            out_pt = final_img

        return self.post_convert(out_pt, img)


class SelectBestEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.VoteEnsemble`.
    """

    backend = SelectBestEnsemble.backend

    def __init__(
        self,
        candidate_per_tissue_path: Path,
        tissue_dict: Dict[str, int],
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
            candidate_per_tissue_path=candidate_per_tissue_path,
            tissue_dict=tissue_dict,
        )
        super().__init__(keys, ensemble, output_key)
