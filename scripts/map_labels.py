import os
import numpy as np
import itk
import typer
import json
from pathlib import Path
from typing import Union

import segmantic
from segmantic.prepro.labels import (
    load_tissue_list,
    save_tissue_list,
    build_tissue_mapping,
)


drcmr_labels_16 = [
    "Background",
    "Air_internal",
    "Artery",
    "Bone_cancellous",
    "Bone_cortical",
    "Cerebrospinal_fluid",
    "Cerebrum_grey_matter",
    "Cerebrum_white_matter",
    "Eyes",
    "Mucosa",
    "Other_tissues",
    "Rectus_muscles",
    "SAT",
    "Skin",
    "Spinal_cord",
    "Vein",
    "Visual_nerve",
]


def premap(name: str):
    return "Other_tissues" if "SAT" == name else name


def map_bone_fg_bg(name: str):
    if name.startswith("Bone_"):
        return "Bone"
    elif name == "Background":
        return "Background"
    return "Head"


def map_bone_skin_air_fg_bg(name: str):
    if name.startswith("Bone_"):
        return "Bone"
    elif name == "Air_internal":
        return "Air_internal"
    elif name == "Skin":
        return "Skin"
    elif name == "Background":
        return "Background"
    return "Head"


def map_vessels2other(name: str):
    if name.startswith("Bone_"):
        return "Bone"
    elif "Vein" == name or "Artery" == name:
        return "Other_tissues"
    return premap(name)


def main(
    input_dir: Path,
    output_dir: Path,
    input_tissues: Path,
    input2output: str,
):
    """Map labels in all nifty files in specified directory

    Args:
        input_dir (Path): input_dir
        output_dir (Path): output_dir
        input_tissues (Path): output tissue list file
        input2output (Path): mapping name [map_bone_fg_bg, map_bone_skin_air_fg_bg, map_vessels2other], or json file mapping input to output tissues

    The json file can be generated using:
        with open(input2output, 'w') as f:
            json.dump({ "Skull": "Bone", "Mandible": "Bone", "Fat": "Fat" }, f)
    """

    # get input and output tissue lists
    if input_tissues:
        imap = load_tissue_list(input_tissues)
    else:
        imap = {n: i for i, n in enumerate(drcmr_labels_16)}

    if os.path.exists(input2output):
        with open(input2output) as f:
            i2omap = json.load(f)
        mapper = lambda n: i2omap[n]
    elif input2output in locals():
        mapper = locals()[str(input2output)]
    else:
        raise RuntimeError("Invalid mapping function specified")

    # build index mapping from input to output
    omap, i2o = build_tissue_mapping(imap, mapper)

    os.makedirs(output_dir, exist_ok=True)
    save_tissue_list(omap, output_dir / "labels_5.txt")

    for input_file in input_dir.glob("*.nii.gz"):
        image = segmantic.imread(input_file)
        image_view = itk.array_view_from_image(image)
        image_view[:] = i2o[image_view[:]]

        assert len(np.unique(image)) == np.max(image) + 1

        segmantic.imwrite(image, output_dir / input_file.name)


if __name__ == "__main__":
    typer.run(main)
