# import h5py
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional, Union
from ..prepro.labels import load_tissue_list, load_tissue_colors


def export_to_iseg(
    iseg_file_path, label_field: np.ndarray, image: np.ndarray, tissuelist_path: Path
):
    labels, colors = load_tissue_list(tissuelist_path, load_colors=True)
    with h5py.File(iseg_file_path) as f:
        pass


class iSegSaver:
    def __init__(
        self,
        output_dir: Union[Path, str] = "./",
        output_postfix: str = "seg",
        output_ext: str = ".h5",
        # resample: bool = True,
        # mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        # padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        # align_corners: bool = False,
        # dtype: DtypeLike = np.float64,
        # output_dtype: DtypeLike = np.float32,
        # squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        pass

    def save(
        self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None
    ) -> None:
        pass

    def save_batch(
        self,
        batch_data: Union[torch.Tensor, np.ndarray],
        meta_data: Optional[Dict] = None,
    ) -> None:
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(
                data=data,
                meta_data={k: meta_data[k][i] for k in meta_data}
                if meta_data is not None
                else None,
            )
