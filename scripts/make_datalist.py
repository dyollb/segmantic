import json
import random
from pathlib import Path

import typer

from segmantic.image.labels import load_tissue_list
from segmantic.seg.dataset import PairedDataSet
from segmantic.utils.file_iterators import find_matching_files

app = typer.Typer()


@app.command()
def make_datalist(
    data_dir: Path = typer.Option(
        ...,
        help="root data directory. Paths in datalist will be relative to this directory",
    ),
    image_dir: Path = typer.Option(..., help="Directory containing images"),
    labels_dir: Path = typer.Option(None, help="Directory containing labels"),
    datalist_path: Path = typer.Option(..., help="Filename of output datalist"),
    num_channels: int = 1,
    num_classes: int = -1,
    tissuelist_path: Path = None,
    percent: float = 1.0,
    description: str = "",
    image_glob: str = "*.nii.gz",
    labels_glob: str = "*.nii.gz",
    test_only: bool = False,
    seed: int = 104,
) -> int:
    # add labels
    if tissuelist_path is not None:
        tissuelist = load_tissue_list(tissuelist_path)
        labels = {str(id): n for n, id in tissuelist.items() if id != 0}
    elif num_classes > 0:
        labels = {str(id): f"tissue{id:02d}" for id in range(1, num_classes + 1)}
    else:
        raise ValueError("Either specify 'tissuelist_path' or 'num_classes'")

    data_config = {
        "description": description,
        "num_channels": num_channels,
        "labels": labels,
    }

    # add all files as test files
    if test_only:
        test_files = (data_dir / image_dir).glob(image_glob)
        data_config["training"] = []
        data_config["validation"] = []
        data_config["test"] = [str(f.relative_to(data_dir)) for f in test_files]

    # build proper datalist with training/validation/test split
    else:
        matches = find_matching_files(
            [data_dir / image_dir / image_glob, data_dir / labels_dir / labels_glob]
        )
        pairs = [
            (p[0].relative_to(data_dir), p[1].relative_to(data_dir)) for p in matches
        ]

        random.Random(seed).shuffle(pairs)
        test, pairs = pairs[:10], pairs[10:]
        num_valid = int(percent * 0.2 * len(pairs))
        num_training = len(pairs) - num_valid if percent >= 1.0 else 4 * num_valid

        data_config["training"] = [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[:num_training]
        ]
        data_config["validation"] = [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[-num_valid:]
        ]
        data_config["test"] = ([str(im) for im, _ in test],)

    return datalist_path.write_text(json.dumps(data_config, indent=2))


@app.command()
def extend_datalist(
    data_dir: Path = typer.Option(
        ...,
        help="root data directory. Paths in datalist will be relative to this directory",
    ),
    image_dir: Path = typer.Option(..., help="Directory containing images"),
    labels_dir: Path = typer.Option(None, help="Directory containing labels"),
    datalist_path: Path = typer.Option(..., help="Filename of input datalist"),
    output_path: Path = typer.Option(..., help="Filename of output datalist"),
    image_glob: str = "*.nii.gz",
    labels_glob: str = "*.nii.gz",
):
    ds = PairedDataSet.load_from_json(datalist_path)
    images = [d["image"].name.lower() for d in ds.training_files()]
    images += [d["image"].name.lower() for d in ds.validation_files()]
    images += [d["image"].name.lower() for d in ds.test_files()]

    if image_dir.is_absolute():
        image_dir = image_dir.relative_to(data_dir)
    if labels_dir.is_absolute():
        labels_dir = labels_dir.relative_to(data_dir)

    matches = find_matching_files(
        [data_dir / image_dir / image_glob, data_dir / labels_dir / labels_glob]
    )

    training_data = list(ds.training_files())
    for p in matches:
        image_name = p[0].name.lower()
        if image_name not in images:
            training_data.append({"image": p[0], "label": p[1]})

    def make_relative(d: dict[str, Path]):
        return {key: str(d[key].relative_to(data_dir)) for key in d}

    data_config = json.loads(datalist_path.read_text())

    data_config["training"] = [make_relative(v) for v in training_data]
    data_config["validation"] = [make_relative(v) for v in ds.validation_files()]
    # data_config["test"] = (make_relative(v)["image"] for v in ds.test_files())
    return output_path.write_text(json.dumps(data_config, indent=2))


@app.command()
def print_stats(datalist: Path):
    ds = PairedDataSet.load_from_json(datalist)
    print(f"Training cases: {len(ds.training_files())}")
    print(f"Validation cases: {len(ds.validation_files())}")
    print(f"Test cases: {len(ds.test_files())}")


def main():
    app()


if __name__ == "__main__":
    main()
