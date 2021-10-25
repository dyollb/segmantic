import itk
import typer
from pathlib import Path

from segmantic.prepro.core import resample_to_ref


def main(input: Path, ref: Path, output: Path) -> None:
    input_img = itk.imread(f"{input}")
    ref_img = itk.imread(f"{ref}")
    out_img = resample_to_ref(input_img, ref_img)
    itk.imwrite(out_img, f"{output}")


if __name__ == "__main__":
    typer.run(main)
