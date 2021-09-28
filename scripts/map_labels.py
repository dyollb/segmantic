import os
from typing import Callable
import numpy as np
import itk
import argparse
from pathlib import Path

from segmantic.prepro.labels import load_tissue_list, save_tissue_list, build_tissue_mapping


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map labels.')
    parser.add_argument('-i', '--input_dir', dest='input_dir', type=Path, required=True, help='input directory')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=Path, required=True, help='output directory')
    parser.add_argument('--input_tissues', dest='input_tissues', type=Path, help='input tissue list file')
    parser.add_argument('--output_tissues', dest='output_tissues', type=Path, required=True, help='output tissue list file')
    args = parser.parse_args()

    # get input and output tissue lists
    if args.input_tissues:
        imap = load_tissue_list(args.input_tissues)
    else:
        imap = {n: i for i, n in enumerate(drcmr_labels_16)}

    if os.path.exists(args.output_tissues):
        omap = load_tissue_list(args.output_tissues)
        def map_name(n: str) -> str:
            return n#omap[n]
        mapper = map_name
    elif args.output_tissues in locals():
        mapper = locals()[args.output_tissues]
    else:
        raise RuntimeError("Invalid mapping function specified")

    # build index mapping from input to output
    omap, i2o = build_tissue_mapping(imap, mapper)

    os.makedirs(args.output_dir, exist_ok=True)
    save_tissue_list(omap, Path(args.output_dir) / "labels_5.txt")

    for f in os.listdir(args.input_dir):
        if not f.endswith(".nii.gz"):
            continue

        image = itk.imread(os.path.join(args.input_dir, f))
        image_view = itk.array_view_from_image(image)
        image_view[:] = i2o[image_view[:]]

        assert len(np.unique(image)) == np.max(image) + 1

        itk.imwrite(image, os.path.join(args.output_dir, f))
