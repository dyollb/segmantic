import os
import shutil
from pathlib import Path


def copy_image_labels(
    dirA: Path, dirB: Path, outdirA: Path, outdirB: Path, suffix: str
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
        dirA=Path(r"F:\Data\DRCMR-Thielscher\mdix0_images"),
        dirB=Path(r"F:\Data\DRCMR-Thielscher\all_data\labels_16"),
        outdirA=Path(r"F:\Data\DRCMR-Thielscher\all_data\mixed_images"),
        outdirB=Path(r"F:\Data\DRCMR-Thielscher\all_data\mixed_labels"),
        suffix="_mdix0",
    )
    copy_image_labels(
        dirA=Path(r"F:\Data\DRCMR-Thielscher\mdix1_images"),
        dirB=Path(r"F:\Data\DRCMR-Thielscher\all_data\labels_16"),
        outdirA=Path(r"F:\Data\DRCMR-Thielscher\all_data\mixed_images"),
        outdirB=Path(r"F:\Data\DRCMR-Thielscher\all_data\mixed_labels"),
        suffix="_mdix1",
    )
