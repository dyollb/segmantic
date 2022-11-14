from pathlib import Path

import SimpleITK as sitk
import typer

from segmantic.image.processing import resample_to_ref


def main(input: Path, ref: Path, output: Path, nearest: bool) -> None:
    input_img = sitk.ReadImage(input)
    ref_img = sitk.ReadImage(ref)
    out_img = resample_to_ref(input_img, ref_img, nearest=nearest)
    sitk.WriteImage(out_img, output)


if __name__ == "__main__":
    typer.run(main)
