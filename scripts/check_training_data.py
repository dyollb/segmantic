from pathlib import Path

import numpy as np
import SimpleITK as sitk
import typer

from segmantic.utils.file_iterators import find_matching_files


def check_training_data(
    image_dir: Path, labels_dir: Path, copy_image_information: bool = False
):
    matches = find_matching_files([image_dir / "*.nii.gz", labels_dir / "*.nii.gz"])
    for p in matches:
        img = sitk.ReadImage(p[0])
        lbl = sitk.ReadImage(p[1])
        if img.GetSize() != lbl.GetSize():
            print(f"Size mismatch {p[0].name}: {img.GetSize()} != {lbl.GetSize()}")
            continue
        if copy_image_information:
            lbl.CopyInformation(img)
            sitk.WriteImage(sitk.Cast(lbl, sitk.sitkUInt8), p[1])
        elif img.GetSpacing() != lbl.GetSpacing() or img.GetOrigin() != lbl.GetOrigin():
            np.testing.assert_almost_equal(
                img.GetSpacing(), lbl.GetSpacing(), decimal=2
            )
            np.testing.assert_almost_equal(img.GetOrigin(), lbl.GetOrigin(), decimal=2)


if __name__ == "__main__":
    typer.run(check_training_data)
