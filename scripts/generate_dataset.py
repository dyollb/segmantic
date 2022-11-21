import json
from pathlib import Path

import typer

from segmantic.seg.dataset import find_matching_files
from segmantic.utils.json import PathEncoder


def generate_dataset(
    *,
    image_dir: Path,
    labels_dir: Path,
    image_glob: str = "*.nii.gz",
    labels_glob: str = "*.nii.gz",
    dataset_file: Path = None,
):
    """Generate json file describing"""
    data = find_matching_files([image_dir / image_glob, labels_dir / labels_glob])

    data_dicts = [{"image": p[0], "label": p[1]} for p in data]

    json_text = json.dumps(
        {
            "training": data_dicts,
        },
        cls=PathEncoder,
    )
    if dataset_file:
        dataset_file.write_text(json_text)
    else:
        print(json_text)


if __name__ == "__main__":
    typer.run(generate_dataset)
