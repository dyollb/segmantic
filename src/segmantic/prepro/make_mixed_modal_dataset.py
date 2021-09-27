import os
import shutil
from pathlib import Path
from typing import Callable, List, Union

from .core import get_files, PathLike


def copy_image_labels(
    dirA: PathLike, dirB: PathLike, outdirA: PathLike, outdirB: PathLike, suffix: str
) -> None:
    b_files = list(os.listdir(dirB))
    for f in os.listdir(dirA):
        if f in b_files:
            shutil.copy(
                src=os.path.join(dirA, f),
                dst=os.path.join(outdirA, f.replace(".nii.gz", suffix + ".nii.gz")),
            )
            shutil.copy(
                src=os.path.join(dirB, f),
                dst=os.path.join(outdirB, f.replace(".nii.gz", suffix + ".nii.gz")),
            )


if __name__ == "__main__":
    copy_image_labels(
        dirA=r"F:\Data\DRCMR-Thielscher\mdix0_images",
        dirB=r"F:\Data\DRCMR-Thielscher\all_data\labels_16",
        outdirA=r"F:\Data\DRCMR-Thielscher\all_data\mixed_images",
        outdirB=r"F:\Data\DRCMR-Thielscher\all_data\mixed_labels",
        suffix="_mdix0",
    )
    copy_image_labels(
        dirA=r"F:\Data\DRCMR-Thielscher\mdix1_images",
        dirB=r"F:\Data\DRCMR-Thielscher\all_data\labels_16",
        outdirA=r"F:\Data\DRCMR-Thielscher\all_data\mixed_images",
        outdirB=r"F:\Data\DRCMR-Thielscher\all_data\mixed_labels",
        suffix="_mdix1",
    )
