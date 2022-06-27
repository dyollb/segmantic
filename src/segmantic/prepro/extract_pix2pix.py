import os
from pathlib import Path
from random import randint

import itk
import numpy as np

from .core import NdarrayImage, identity, imread, imwrite
from .modality import scale_clamp_ct


def export_slices(  # type: ignore
    im1_dir: Path,
    im2_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    tag1: str = "A",
    tag2: str = "B",
    process_img1=identity,
    process_img2=identity,
) -> None:
    """Create paired dataset for use with pix2pix (first run combine_A_B.py)"""
    files = []
    for p1 in im1_dir.glob("*.nii.gz"):
        p2 = im2_dir / p1.name
        p3 = labels_dir / p1.name
        if p2.exists() and p3.exists():
            files.append((p1, p2, p3))

    # create folders for output
    folder = ["train"] * max(len(files) - 6, 1) + ["val"] * 3 + ["test"] * 3
    for sub in ["train", "val", "test"]:
        os.makedirs(output_dir / tag1 / sub, exist_ok=True)
        os.makedirs(output_dir / tag2 / sub, exist_ok=True)

    # loop over 3d images
    for idx, (p1, p2, p3) in enumerate(files):
        img1 = process_img1(imread(p1))
        img2 = process_img2(imread(p2))
        labels = imread(p3)
        base = p1.name

        print(np.min(img1), np.max(img1))
        print(np.min(img2), np.max(img2))

        # for each slice
        for k in range(labels.shape[0]):
            if np.max(labels[k, :, :]) == 0:
                continue

            s1 = randint(0, img1.shape[1] - 256)
            s2 = randint(0, img1.shape[2] - 256)
            slice1 = itk.image_from_array(img1[k, s1 : s1 + 256, s2 : s2 + 256])
            slice2 = itk.image_from_array(img2[k, s1 : s1 + 256, s2 : s2 + 256])
            imwrite(
                slice1.astype(itk.SS),
                output_dir
                / tag1
                / folder[idx]
                / base.replace(".nii.gz", "_%03d.tif" % k),
                compression=True,
            )
            imwrite(
                slice2.astype(itk.SS),
                output_dir
                / tag2
                / folder[idx]
                / base.replace(".nii.gz", "_%03d.tif" % k),
                compression=True,
            )


def preprocess_mri(x: NdarrayImage) -> NdarrayImage:
    x_view = itk.array_view_from_image(x)
    x_view *= 255.0 / 280.0
    np.clip(x_view, a_min=0, a_max=255, out=x_view)
    return x


if __name__ == "__main__":

    # import cv2
    # im = cv2.imread(r"F:\Temp\t12ct\ct\train\X98777_047.tif", cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    #
    # from PIL import Image
    # im = Image.open(r"F:\Temp\t12ct\ct\train\X98777_047.tif")
    # pix = np.array(im)
    # print(pix.shape, pix.dtype)

    export_slices(
        im1_dir=Path(r"F:\Data\DRCMR-Thielscher\all_data\images"),  # T1
        im2_dir=Path(r"F:\Data\DRCMR-Thielscher\for_machine_learning\images"),  # CT
        labels_dir=Path(r"F:\Data\DRCMR-Thielscher\all_data\labels_16"),
        output_dir=Path(r"F:\temp\t1w2ctm"),
        tag1="t1w",
        tag2="ct",
        process_img1=preprocess_mri,
        process_img2=scale_clamp_ct,
    )
