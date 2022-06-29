import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import typer


def fix_binary_masks(directory: Path, file_glob: str = "*.nii.gz"):
    logger = logging.getLogger(__file__)
    for file_path in directory.glob(file_glob):
        img = nib.load(f"{file_path}")
        data = img.get_fdata()
        max_value = np.max(data)
        if max_value == 0:
            logger.error("%s mask is empty", file_path)
            return

        min_value = np.min(data[data != 0])
        if min_value < 1 or max_value != 1:
            mask = np.zeros_like(data, dtype=np.uint8)
            mask[data > 0.5] = 1
            nib.save(nib.Nifti1Image(mask, img.affine), f"{file_path}")
            logger.warning(
                "%s foreground values in range [%s,%s]",
                file_path,
                f"{min_value}",
                f"{max_value}",
            )


if __name__ == "__main__":
    typer.run(fix_binary_masks)
