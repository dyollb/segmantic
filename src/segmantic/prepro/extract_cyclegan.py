import os
import numpy as np
import itk
from random import randint
from typing import List, Tuple
from pathlib import Path

from .core import (
    extract_slices,
    resample,
    scale_to_range,
    identity,
    get_files,
    AnyImage,
    Image2,
)
from .modality import scale_clamp_ct


def bbox(img: Image2) -> Tuple[float, float, float, float]:
    """Get foreground (non-zero) bounding box from 2D image"""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def export_slices(  # type: ignore
    image_files: List[Path],
    output_dir: Path,
    axis: int = 2,
    flip_lr: bool = False,
    process_img=identity,
) -> None:
    """Load list of 3D images and extract & export slices

    Args:
        image_files (List[Path]): List of 3D image files to process
        output_dir (Path): Slices will be exported to this folder
        axis (int, optional): The axis defining the slice. Defaults to 2 (XY)
        flip_lr (bool, optional): Flip slices. Defaults to False.
        process_img ([type], optional): A user-defined callback to process the 3D image. Defaults to identity.
    """
    # create folders for output
    os.makedirs(output_dir, exist_ok=True)

    # loop over 3d images
    for file_path in image_files:
        img = itk.imread(file_path)

        slices = extract_slices(img=process_img(img), axis=axis)
        f = os.path.basename(file_path)

        # for each slice
        for k, slice in enumerate(slices):

            # skip if 'empty'
            if np.percentile(slice, 98) < 10:
                continue

            # pad to have minimum width 256, or 300, ...
            pad_size = 300
            if slice.shape[0] < pad_size or slice.shape[1] < pad_size:
                sz = slice.shape
                pad_x = (pad_size + 1 - min(sz[0], pad_size)) // 2
                pad_y = (pad_size + 1 - min(sz[1], pad_size)) // 2
                slice = np.pad(
                    slice,
                    ((pad_x, pad_x), (pad_y, pad_y)),
                    mode="constant",
                    constant_values=0,
                )

            # random crop
            if slice.shape[0] > 256 or slice.shape[1] > 256:

                rmin, rmax, cmin, cmax = bbox(slice > 4)
                s1 = randint(0, min(rmin, max(0, slice.shape[0] - 256)))
                s2 = randint(0, min(cmin, max(0, slice.shape[1] - 256)))
                slice = slice[s1 : s1 + 256, s2 : s2 + 256]

            # flip?
            if flip_lr:
                slice = np.rot90(slice, k=1)

            itk.imwrite(
                itk.image_from_array(slice).astype(itk.SS),
                output_dir / f.replace(".nii.gz", "_%03d.tif" % k),
                compression=True,
            )


def scale_to_uchar(img: AnyImage) -> AnyImage:
    return scale_to_range(img, vmin=0, vmax=255)


def convert_to_rgb(files: List[Path]) -> None:
    from PIL import Image

    for f in files:
        im = Image.open(f)
        imgc = im.convert("RGB")
        imgc.save(f)
        im.close()
        imgc.close()


def randomize_files(dir: Path, ext: str = ".tif") -> None:
    """randomize files in a folder

    Args:
        dir (Path): the directory to search in
        ext (str, optional): the file extension. Defaults to ".tif".
    """
    import shutil
    from random import sample

    random_sample = lambda x: sample(x, len(x))

    files = random_sample([f for f in dir.glob("*%s" % ext)])
    for i, f in enumerate(files):
        shutil.move(f, dir / ("im_%05d.tif" % i))


if __name__ == "__main__":

    t1_images = get_files(Path(r"F:\Data\DRCMR-Thielscher\all_data\images"))
    ixi_t1_images = get_files(
        Path(r"C:\Users\lloyd\Downloads\IXI-T1"), predicate=lambda x: "Guys" in x
    )

    # export_slices(
    #    image_files=t1_images[:-6],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\trainA",
    #    process_img=scale_to_uchar,
    # )
    # export_slices(
    #    image_files=t1_images[-6:],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\testA",
    #    process_img=scale_to_uchar,
    # )

    # export_slices(
    #    image_files=ixi_t1_images[:15],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\trainB",
    #    process_img=lambda x: resample(scale_to_uchar(x)),
    #    axis=1,
    #    flip_lr=True,
    # )
    # convert_to_rgb(get_files(r"F:\temp\cyclegan\t1_drcmr2ixi\trainB", ext=".tif"))

    export_slices(
        image_files=ixi_t1_images[15:21],
        output_dir=Path(r"F:\temp\cyclegan\t1_drcmr2ixi\testB"),
        process_img=lambda x: resample(scale_to_uchar(x)),
        axis=1,
        flip_lr=True,
    )
    convert_to_rgb(
        get_files(
            Path(r"F:\temp\cyclegan\t1_drcmr2ixi\testB"),
            predicate=lambda x: x.endswith(".tif"),
        )
    )
    randomize_files(Path(r"F:\temp\cyclegan\t1_drcmr2ixi\testB"))
