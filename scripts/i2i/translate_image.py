import numpy as np
import itk
import typer
from pathlib import Path
from typing import List

import segmantic
from segmantic.prepro.core import crop_center
from segmantic.i2i.translate import load_pix2pix_generator, translate_3d
from segmantic.seg.utils import make_device


def preprocess_mri(x):
    x_view = itk.array_view_from_image(x)
    x_mean, x_std = np.mean(x_view), np.std(x_view)
    x_view -= x_mean
    x_view *= 0.5 / x_std
    x_view += 0.5
    # np.clip(x_view, a_min=0, a_max=1, out=x_view)
    return x


def main(
    model: Path = typer.Option(..., "--model", "-m", help="model file path"),
    input: Path = typer.Option(..., "--input", "-i", help="input image file"),
    output: Path = typer.Option("out.nii.gz", "--output", "-o"),
    axis: int = 2,
    debug_axis: bool = False,
    gpu_ids: List[int] = [],
):
    """Translate image using style transfer model"""

    if not input.exists():
        raise ValueError(f"Invalid input {input}")

    if not output.parent.exists():
        raise ValueError(f"Expected output dir {output.parent}")

    if debug_axis:
        crop_size = [1024, 1024, 1024]
        crop_size[axis] = 1
        segmantic.imwrite(
            crop_center(segmantic.imread(input), target_size=crop_size), output
        )
        return

    # TODO: resample/pad
    preprocess = lambda img: crop_center(img, target_size=(256, 256, 10))
    postprocess = lambda img: img

    device = make_device(gpu_ids)

    # load model
    netg = load_pix2pix_generator(model_file_path=model, device=device, eval=False)

    # load input image
    img_t1 = segmantic.imread(input)

    # translate slice-by-slice
    img_ct = translate_3d(preprocess(img_t1), model=netg, axis=axis, device=device)

    # write translated image
    segmantic.imwrite(postprocess(img_ct), output)


if __name__ == "__main__":
    typer.run(main)
