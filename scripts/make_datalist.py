import json
import random
from pathlib import Path
from typing import List

import typer

from segmantic.image.labels import load_tissue_list


def find_matching_files(input_globs: List[Path], verbose: bool = True):
    dir_0 = Path(input_globs[0].anchor)
    glob_0 = str(input_globs[0].relative_to(dir_0))
    ext_0 = input_globs[0].name.rsplit("*")[-1]

    candidate_files = {p.name.replace(ext_0, ""): [p] for p in dir_0.glob(glob_0)}

    for other_glob in input_globs[1:]:
        dir_i = Path(other_glob.anchor)
        glob_i = str(other_glob.relative_to(dir_i))
        ext_i = other_glob.name.rsplit("*")[-1]

        for p in dir_i.glob(glob_i):
            key = p.name.replace(ext_i, "")
            if key in candidate_files:
                candidate_files[key].append(p)
            elif verbose:
                print(f"No match found for {key} : {p}")

    output_files = [v for v in candidate_files.values() if len(v) == len(input_globs)]

    if verbose:
        print(f"Number of files in {input_globs[0]}: {len(candidate_files)}")
        print(f"Number of tuples: {len(output_files)}\n")

    return output_files


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
        pairs = find_matching_files(
            [data_dir / image_dir / image_glob, data_dir / labels_dir / labels_glob]
        )
        pairs = [
            (im.relative_to(data_dir), lbl.relative_to(data_dir)) for im, lbl in pairs
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
