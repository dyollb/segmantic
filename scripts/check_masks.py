from pathlib import Path

import nibabel as nib
import numpy as np
import typer


def check_masks(directory: Path, file_glob: str = "*.nii.gz"):
    for f in directory.glob(file_glob):
        img = nib.load(f"{f}")
        data = img.get_fdata()
        max_value = np.max(data)
        if max_value == 0:
            print(f"ERROR: {f} mask is empty")
            return

        min_value = np.min(data[data != 0])
        if min_value < 1 or max_value != 1:
            mask = np.zeros_like(data, dtype=np.uint8)
            mask[data > 0.5] = 1
            nib.save(nib.Nifti1Image(mask, img.affine), f"{f}")
            print(f"WARNING: {f} foreground values in range [{min_value}, {max_value}]")


if __name__ == "__main__":
    typer.run(check_masks)
