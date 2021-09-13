import os
import numpy as np
import itk
import argparse

from segmania.prepro.labels import save_tissue_list, build_map


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
    parser.add_argument('-i', '--input_dir', dest='in_dir', type=str, required=True, help='input directory')
    parser.add_argument('-o', '--output_dir', dest='out_dir', type=str, help='output directory')
    args = parser.parse_args()

    imap, omap, i2o = build_map(drcmr_labels_16, map_bone_skin_air_fg_bg)

    os.makedirs(args.out_dir, exist_ok=True)
    save_tissue_list(omap, os.path.join(args.out_dir, "labels_5.txt"))

    for f in os.listdir(args.in_dir):
        if not f.endswith(".nii.gz"):
            continue

        image = itk.imread(os.path.join(args.in_dir, f))
        image[:] = i2o[image[:]]

        assert len(np.unique(image)) == np.max(image) + 1

        itk.imwrite(image, os.path.join(args.out_dir, f))
