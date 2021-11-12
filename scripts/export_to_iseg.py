import numpy as np
import h5py
import itk
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import typer


def load_tissue_list(file_name: Path, load_colors: bool = False):
    """Load tissue list in iSEG format

    Example file:
        V7
        N3
        C0.00 0.00 1.00 0.50 Bone
        C0.00 1.00 0.00 0.50 Fat
        C1.00 0.00 0.00 0.50 Skin
    """
    tissue_label_map = {"Background": 0}
    tissue_color_map = {0: (0.0, 0.0, 0.0)}
    next_id = 1
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith("C"):
                tissue = line.rsplit(" ", 1)[-1].rstrip()
                rgba = [float(v.strip()) for v in line.lstrip("C").split(" ")[:-1]]
                if tissue in tissue_label_map:
                    raise KeyError(f"duplicate label '{tissue}' found in '{file_name}'")
                tissue_label_map[tissue] = next_id
                tissue_color_map[next_id] = (rgba[0], rgba[1], rgba[2])
                next_id += 1
    if not load_colors:
        return tissue_label_map
    return tissue_label_map, tissue_color_map


def extract_first_component(img):
    image_view = itk.array_view_from_image(img)
    if len(image_view.shape) == 4:
        arr_c0 = image_view[:, :, :, 0]
        img_c0 = itk.image_from_array(arr_c0)
        img_c0.SetOrigin(itk.origin(img))
        img_c0.SetSpacing(itk.spacing(img))
        img_c0.SetDirection(img.GetDirection())
        return img_c0
    return img


def resample_to_ref(img, ref):
    """resample (2D/3D) image to a reference grid

    Args:
        img: input image
        ref: reference image

    Returns:
        itkImage: resampled image
    """
    dim = img.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    transform = itk.IdentityTransform[itk.D, dim].New()

    # resample to target resolution
    resampled = itk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=itk.size(ref),
        output_spacing=itk.spacing(ref),
        output_origin=itk.origin(ref),
        output_direction=ref.GetDirection(),
    )
    return resampled


def convert_to_iseg(
    iseg_file_path: Path,
    label_field_path: Path,
    image_path: Path,
    tissuelist_path: Path,
):
    label_field = itk.imread(f"{label_field_path}")
    image = resample_to_ref(
        extract_first_component(itk.imread(f"{image_path}")), label_field
    )
    labels, colors = load_tissue_list(tissuelist_path, load_colors=True)

    label_field_view = itk.array_view_from_image(label_field)
    image_view = itk.array_view_from_image(image)

    max_label = int(np.max(label_field_view))
    for i in range(max(labels.values()) + 1, max_label + 1):
        labels[f"tissue_{i}"] = i

    dimensions = [s for s in itk.size(image)]
    spacing = [s for s in itk.spacing(image)]
    origin = [o for o in itk.origin(image)]
    dir = image.GetDirection()

    rot = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rot[i, j] = dir(i, j)

    with h5py.File(iseg_file_path, "w") as fout:
        fout.create_dataset(
            "Tissue",
            dtype=np.uint16,
            data=label_field_view.flatten(),
            compression="gzip",
            compression_opts=1,
        )
        fout.create_dataset(
            "Source",
            dtype=np.float32,
            data=image_view.flatten(),
            compression="gzip",
            compression_opts=1,
        )
        fout.create_dataset(
            "Target",
            dtype=np.float32,
            data=np.zeros(image_view.flatten().shape),
            compression="gzip",
            compression_opts=1,
        )
        fout.create_dataset("rotation", dtype=np.float32, data=rot.flatten())
        fout.create_dataset("dimensions", dtype=int, data=dimensions)
        fout.create_dataset("offset", dtype=np.float32, data=origin)
        fout.create_dataset("pixelsize", dtype=np.float32, data=spacing)

        tissues = fout.create_group("Tissues")
        for k in labels.keys():
            try:
                idx = labels[k]
                rgbo = 0.5 * np.ones((4,), dtype=np.float32)
                if idx in colors:
                    rgbo[0] = colors[idx][0]
                    rgbo[1] = colors[idx][1]
                    rgbo[2] = colors[idx][2]
                T = tissues.create_group(k)
                T.create_dataset(
                    "index", dtype=int, data=np.array([idx], dtype=np.int32)
                )
                T.create_dataset("rgbo", dtype=np.float32, data=rgbo)
            except:
                print("Problems writing: %s" % k)
        tissues.create_dataset(
            "bkg_rgbo", dtype=np.float32, data=np.array([0, 0, 0, 0.5])
        )
        tissues.create_dataset("version", dtype=int, data=np.array([0], dtype=int))


if __name__ == "__main__":
    typer.run(convert_to_iseg)
