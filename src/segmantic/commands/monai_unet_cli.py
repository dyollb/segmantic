import inspect
import json
from functools import partial
from pathlib import Path
from typing import List

import typer
import yaml

from ..prepro.labels import load_tissue_list
from ..seg import monai_unet
from ..util.encoders import PathEncoder

app = typer.Typer()


def _is_path(param: inspect.Parameter) -> bool:
    if param.annotation != inspect.Parameter.empty and inspect.isclass(
        param.annotation
    ):
        return issubclass(param.annotation, Path)
    return False


def _get_dumper():
    def _path_representer(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

    dumper = yaml.Dumper
    dumper.add_representer(Path, _path_representer)
    return dumper


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

    The config should either specify a 'dataset' or an 'image_dir'/'labels_dir' pair.

    The dataset can be a single file or a list of files in json format, specifying
    lists of image and label files, or glob expressions for image and labels.

    Example config using 'image_dir'/'labels_dir':
    {
        "image_dir" = "/dataA/image",
        "labels_dir" = "/dataA/label",
        "output_dir" = "<path where trained model and logs are saved>",
        ...
    }

    Example config 'single dataset':
    {
        "dataset" = "/dataA/dataset.json",
        "output_dir" = "<path where trained model and logs are saved>",
        ...
    }

    Example config 'multiple datasets':
    {
        "dataset" = ["/dataA/dataset.json", "/dataB/dataset.json"],
        "output_dir" = "<path where trained model and logs are saved>",
        ...
    }
    """
    sig = inspect.signature(monai_unet.train)

    format_is_json = config_file and config_file.suffix.lower() == ".json"
    dumps = (
        partial(json.dumps, cls=PathEncoder, indent=4)
        if format_is_json
        else partial(yaml.dump, Dumper=_get_dumper(), sort_keys=False)
    )
    loads = json.loads if format_is_json else yaml.safe_load

    if print_defaults:
        cast_from_path = (
            lambda v, k: str(v) if v and _is_path(sig.parameters[k]) else v
        )  # noqa: E731
        default_args = {
            k: cast_from_path(v.default, k)
            if v.default is not inspect.Parameter.empty
            else f"<required option: {v.annotation.__name__}>"
            for k, v in sig.parameters.items()
        }
        if config_file:
            config_file.write_text(dumps(default_args))
        print(dumps(default_args))
        return
    elif not config_file:
        raise ValueError("Invalid '--config-file' argument")

    cast_to_path = (
        lambda v, k: Path(v) if v and _is_path(sig.parameters[k]) else v
    )  # noqa: E731

    args: dict = loads(config_file.read_text())
    args = {k: cast_to_path(v, k) for k, v in args.items()}
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
