from monai.config import print_config
import os
import json
import typer
from pathlib import Path
from typing import List, Optional

from segmantic.prepro.labels import load_tissue_list
from segmantic.seg import monai_unet

app = typer.Typer()


def get_nifti_files(dir: Path) -> List[Path]:
    if not dir:
        return []
    return sorted([f for f in dir.glob("*.nii.gz")])


@app.command()
def train(
    image_dir: Path = typer.Option(
        ..., "--image-dir", "-i", help="directory containing images"
    ),
    labels_dir: Path = typer.Option(
        None, "--labels-dir", "-l", help="directory containing labelfields"
    ),
    tissue_list: Path = typer.Option(
        ..., "--tissue-list", "-t", help="label descriptors in iSEG format"
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        "-r",
        help="output directory where model checkpoints and logs are saved",
    ),
    num_channels: int = 1,
    max_epochs: int = 600,
    gpu_ids: List[int] = [0],
):
    """Train UNet

    Example invocation:

        -i ./dataset/images -l ./dataset/labels --results_dir ./results --tissue_list ./dataset/labels.txt
    """

    print_config()

    tissue_dict = load_tissue_list(tissue_list)
    num_classes = max(tissue_dict.values()) + 1
    if not len(tissue_dict) == num_classes:
        raise ValueError("Expecting contiguous labels in range [0,N-1]")

    os.makedirs(results_dir, exist_ok=True)
    log_dir = Path(results_dir) / "logs"
    model_file = Path(results_dir) / f"drcmr_{num_classes:d}.ckpt"

    monai_unet.train(
        image_dir=image_dir,
        labels_dir=labels_dir,
        log_dir=log_dir,
        num_classes=num_classes,
        num_channels=num_channels,
        model_file_name=model_file,
        max_epochs=max_epochs,
        output_dir=results_dir,
        save_nifti=True,
        gpu_ids=gpu_ids,
    )


@app.command()
def predict(
    image_dir: Path = typer.Option(
        ..., "--image-dir", "-i", help="directory containing images"
    ),
    labels_dir: Path = typer.Option(
        None,
        "--labels-dir",
        "-l",
        help="directory containing labelfields for performance evaluation",
    ),
    model_file: Path = typer.Option(..., "--model-file", "-m", help="saved checkpoint"),
    tissue_list: Path = typer.Option(
        ..., "--tissue-list", "-t", help="label descriptors in iSEG format"
    ),
    results_dir: Path = typer.Option(
        Path("results"), "--results-dir", "-r", help="output directory"
    ),
    gpu_ids: List[int] = [0],
):
    """Predict segmentations

    Example invocation:

        -i ./dataset/images --results_dir ./results --tissue_list ./dataset/labels.txt
    """

    monai_unet.predict(
        model_file=model_file,
        test_images=get_nifti_files(image_dir),
        test_labels=get_nifti_files(labels_dir),
        tissue_dict=load_tissue_list(tissue_list),
        output_dir=results_dir,
        save_nifti=True,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    app()
