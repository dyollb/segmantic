import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import yaml
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.networks import one_hot
from monai.transforms.post.array import Ensemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import Transform
from monai.utils import TransformBackends


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
        num_classes: Optional[int] = None,
    ) -> None:

        self.num_classes = num_classes
        self.tissue_dict = tissue_dict
        with open(f"{candidate_per_tissue_path}") as file:
            self.best_candidate_for_tissue = yaml.full_load(file)

    def __call__(
        self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]
    ) -> NdarrayOrTensor:
        print("in Vote Ensemble __call__")
        img_ = self.get_stacked_torch(img)
        print(torch.Tensor(img_).size()[1:])
        print(img_.size())
        if self.num_classes is not None:
            # has_ch_dim = True
            if img_.ndimension() > 1 and img_.shape[1] > 1:
                warnings.warn("no need to specify num_classes for One-Hot format data.")
            else:
                # if img_.ndimension() == 1:
                # if no channel dim, need to remove channel dim after voting
                # has_ch_dim = False
                img_ = one_hot(img_, self.num_classes, dim=1)

        final_img = torch.empty(img_.size()[1:])

        for tissue_name, tissue_lbl in self.tissue_dict.items():
            best_idx = self.best_candidate_for_tissue[tissue_name]
            final_img[tissue_lbl, :, :, :] = img_[best_idx, tissue_lbl, :, :, :]

        # img_ = torch.mean(img_.float(), dim=0)
        print("after mean of tensor along dim 0")
        print(final_img.size())

        out_pt = torch.argmax(final_img, dim=0, keepdim=True)

        # if self.num_classes is not None:
        # if not One-Hot, use "argmax" to vote the most common class
        #    out_pt = torch.argmax(img_, dim=0, keepdim=has_ch_dim)
        # else:
        # for One-Hot data, round the float number to 0 or 1
        #    out_pt = torch.round(img_)
        print("after finals step")
        print(out_pt.size())
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
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            num_classes: if the input is single channel data instead of One-Hot, we can't get class number
                from channel, need to explicitly specify the number of classes to vote.

        """
        ensemble = SelectBestEnsemble(
            num_classes=num_classes,
            candidate_per_tissue_path=candidate_per_tissue_path,
            tissue_dict=tissue_dict,
        )
        super().__init__(keys, ensemble, output_key)
