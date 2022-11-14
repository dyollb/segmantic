import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import typer

from segmantic.image.labels import (
    build_tissue_mapping,
    load_tissue_list,
    save_tissue_list,
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


def premap(name: str) -> str:
    return "Other_tissues" if "SAT" == name else name


def map_bone_fg_bg(name: str) -> str:
    if name.startswith("Bone_"):
        return "Bone"
    elif name == "Background":
        return "Background"
    return "Head"


def map_bone_skin_air_fg_bg(name: str) -> str:
    if name.startswith("Bone_"):
        return "Bone"
    elif name == "Air_internal":
        return "Air_internal"
    elif name == "Skin":
        return "Skin"
    elif name == "Background":
        return "Background"
    return "Head"


def map_vessels2other(name: str) -> str:
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

        def _map(n: str) -> str:
            return i2omap[n]  # type: ignore

        mapper = _map
    elif input2output in locals():
        mapper = locals()[str(input2output)]
    else:
        raise RuntimeError("Invalid mapping function specified")

    # build index mapping from input to output
    omap, i2o = build_tissue_mapping(imap, mapper)

    os.makedirs(output_dir, exist_ok=True)
    save_tissue_list(omap, output_dir / "labels_5.txt")

    for input_file in input_dir.glob("*.nii.gz"):
        image = sitk.ReadImage(input_file)
        image_np = sitk.GetArrayFromImage(image)
        image_np[:] = i2o[image_np[:]]
        image_mapped = sitk.GetImageFromArray(image_np)
        image_mapped.CopyInformation(image)

        assert len(np.unique(image_np)) == np.max(image_np) + 1

        sitk.WriteImage(image_mapped, output_dir / input_file.name)


if __name__ == "__main__":
    typer.run(main)
