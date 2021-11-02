import json
import inspect
import typer
from pathlib import Path
from typing import List

from segmantic.prepro.labels import load_tissue_list
from segmantic.seg import monai_unet

app = typer.Typer()


def get_nifti_files(dir: Path) -> List[Path]:
    if not dir:
        return []
    return sorted([f for f in dir.glob("*.nii.gz")])


@app.command()
def train_config(
    config_file: Path = typer.Option(
        ..., "--config-file", "-c", help="config file in json format"
    ),
    print_defaults: bool = False,
):
    """Train UNet with configuration provided as a json file

    Example invocation:
        --config-file my_config.json

    To generate a default config:
        --config-file my_config.json --print-defaults
    """
    if print_defaults:
        sig = inspect.signature(monai_unet.train)

        default_args = {
            k: v.default
            if v.default is not inspect.Parameter.empty
            else f"<required option: {v.annotation.__name__}>"
            for k, v in sig.parameters.items()
        }
        with open(config_file, "w") as f:
            json.dump(default_args, f, indent=4)
        return

    with open(config_file, "r") as f:
        args = json.load(f)
        monai_unet.train(**args)


@app.command()
def train(
    image_dir: Path = typer.Option(
        ..., "--image-dir", "-i", help="directory containing images"
    ),
    labels_dir: Path = typer.Option(
        ..., "--labels-dir", "-l", help="directory containing labelfields"
    ),
    tissue_list: Path = typer.Option(
        ..., "--tissue-list", "-t", help="label descriptors in iSEG format"
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        "-r",
        help="output directory where model checkpoints and logs are saved",
    ),
    num_channels: int = 1,
    max_epochs: int = 600,
    gpu_ids: List[int] = [0],
):
    """Train UNet

    Example invocation:
        -i ./dataset/images -l ./dataset/labels --output-dir ./results --tissue_list ./dataset/labels.txt
    """

    monai_unet.train(
        image_dir=image_dir,
        labels_dir=labels_dir,
        tissue_list=tissue_list,
        num_channels=num_channels,
        max_epochs=max_epochs,
        output_dir=output_dir,
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
        help="directory containing labelfields",
    ),
    model_file: Path = typer.Option(
        ..., "--model-file", "-m", help="saved model checkpoint"
    ),
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
