import json
import random
from pathlib import Path

import typer

from segmantic.image.labels import load_tissue_list
from segmantic.utils.file_iterators import find_matching_files


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


def main():
    typer.run(make_datalist)


if __name__ == "__main__":
    main()
