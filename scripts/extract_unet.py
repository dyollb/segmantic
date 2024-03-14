from pathlib import Path
from typing import Optional

import typer


def extract_unet(input_file: Path, output_file: Optional[Path] = None):
    """Load segmantic unet lightning module and export inner monai UNet"""
    import torch

    from segmantic.seg.monai_unet import Net

    if output_file is None:
        output_file = input_file.with_suffix(".pth")
    if output_file.exists() and output_file.samefile(input_file):
        raise RuntimeError("Input and output file are identical")
    net = Net.load_from_checkpoint(input_file)
    torch.save(net._model.state_dict(), output_file)


if __name__ == "__main__":
    typer.run(extract_unet)
