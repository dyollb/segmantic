import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import typer


def fix_binary_masks(directory: Path, file_glob: str = "*.nii.gz"):
    logger = logging.getLogger(__file__)
    for file_path in directory.glob(file_glob):
        img: nib.Nifti1Image = nib.load(f"{file_path}")  # type: ignore [assignment]
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


def round_half_up(input_dir: Path, output_dir: Path = None):
    import SimpleITK as sitk

    for f in input_dir.glob("*.nii.gz"):
        img = sitk.ReadImage(f)
        img_np = sitk.GetArrayViewFromImage(img)
        imin, imax = np.min(img_np), np.max(img_np)
        if imin < 0 or imax > 3:
            print(f"{f.name}: [{imin}, {imax}]")
        if img.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64):
            print(f"{f.name}: {img.GetPixelIDTypeAsString()}")


if __name__ == "__main__":
    typer.run(round_half_up)
