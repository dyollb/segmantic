import json
from pathlib import Path

import typer

from segmantic.util.json import PathEncoder


def generate_dataset(
    *,
    image_dir: Path,
    labels_dir: Path,
    image_glob: str = "*.nii.gz",
    labels_glob: str = "*.nii.gz",
    dataset_file: Path = None,
):
    """Generate json file describing"""

    labels_ext = labels_glob.replace("*", "", 1)
    labels_file = {
        p.name.replace(labels_ext, ""): p for p in labels_dir.glob(labels_glob)
    }

    print(f"Number of label images: {len(labels_file)}")

    image_ext = image_glob.replace("*", "", 1)
    data_dicts = []

    for p in image_dir.glob(image_glob):
        key = p.name.replace(image_ext, "")
        if key in labels_file:
            data_dicts.append({"image": p, "label": labels_file[key]})

    print(f"Number of pairs: {len(data_dicts)}\n")

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
