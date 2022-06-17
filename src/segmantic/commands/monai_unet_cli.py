import json
import inspect
from click import BadArgumentUsage
import typer
from pathlib import Path
from typing import List

from ..prepro.labels import load_tissue_list
from ..seg import monai_unet

app = typer.Typer()


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return f"{obj}"
        return json.JSONEncoder.default(self, obj)


def _is_path(param: inspect.Parameter) -> bool:
    if param.annotation != inspect.Parameter.empty and inspect.isclass(
        param.annotation
    ):
        return issubclass(param.annotation, Path)
    return False


def _get_nifti_files(dir: Path) -> List[Path]:
    if not dir:
        return []
    return sorted([f for f in dir.glob("*.nii.gz")])


@app.command()
def train_config(
    config_file: Path = typer.Option(
        None, "--config-file", "-c", help="config file in json format"
    ),
    print_defaults: bool = False,
) -> None:
    """Train UNet with configuration provided as json file

    Example invocation:

        --config-file my_config.json

    To generate a default config:

        --config-file my_config.json --print-defaults
    """
    sig = inspect.signature(monai_unet.train)

    if print_defaults:
        default_args = {
            k: v.default
            if v.default is not inspect.Parameter.empty
            else f"<required option: {v.annotation.__name__}>"
            for k, v in sig.parameters.items()
        }
        if config_file:
            config_file.write_text(json.dumps(default_args, indent=4, cls=PathEncoder))
        print(json.dumps(default_args, indent=4, cls=PathEncoder))
        return
    elif not config_file:
        raise ValueError("Invalid '--config-file' argument")

    cast_path = lambda v, k: Path(v) if v and _is_path(sig.parameters[k]) else v

    args: dict = json.loads(config_file.read_text())
    args = {k: cast_path(v, k) for k, v in args.items()}
    monai_unet.train(**args)


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
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        "-r",
        help="output directory where model checkpoints and logs are saved",
    ),
    num_channels: int = 1,
    max_epochs: int = 600,
    gpu_ids: List[int] = [0],
) -> None:
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
) -> None:
    """Predict segmentations

    Example invocation:

        -i ./dataset/images -m model.ckpt --results-dir ./results --tissue-list ./dataset/labels.txt
    """

    monai_unet.predict(
        model_file=model_file,
        test_images=_get_nifti_files(image_dir),
        test_labels=_get_nifti_files(labels_dir),
        tissue_dict=load_tissue_list(tissue_list),
        output_dir=results_dir,
        save_nifti=True,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    app()
