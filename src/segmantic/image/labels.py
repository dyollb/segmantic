import colorsys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

RGBTuple = Tuple[float, float, float]


def build_tissue_mapping(
    input_label_map: Dict[str, int], mapper: Callable[[str], str]
) -> Tuple[Dict[str, int], np.ndarray]:
    """Build mapping to map label fields

    Args:
        input_label_map: Dict of tissue names and labels, e.g. loaded with 'load_tissue_list'
        mapper: Gunction that maps a tissue name to a new name

    Returns:
        1. tissue label dict after mapping
        2. input-to-output lookup table
    """
    output_label_names = list(sorted({mapper(n) for n in input_label_map.keys()}))
    output_label_names.remove("Background")
    output_label_names = ["Background"] + output_label_names
    output_label_map = {n: i for i, n in enumerate(output_label_names)}

    input2output = np.zeros((len(input_label_map),), dtype=np.uint16)
    for name in input_label_map.keys():
        index = input_label_map[name]
        name_mapped = mapper(name)
        index_mapped = output_label_map[name_mapped]
        input2output[index] = index_mapped
    return output_label_map, input2output


def save_tissue_list(
    tissue_label_map: Dict[str, int],
    tissue_list_file_name: Path,
    tissue_color_map: Callable[[str], RGBTuple] = None,
) -> None:
    """Save tissue list in iSEG format

    Example:
        from segmantic.image import labels
        labels.save_tissue_list({ 'Bone': 1, 'Fat': 2, 'Skin': 3 }, 'tissues.txt')

    Note:
        Label '0' is implicitly called Background
    """
    # invert dictionary
    num_tissues = max(tissue_label_map.values())
    label_tissue_map = {}
    for name in tissue_label_map.keys():
        label = tissue_label_map[name]
        if label in label_tissue_map:
            raise KeyError("duplicate labels found in 'tissue_label_map'")
        label_tissue_map[label] = name

    if tissue_color_map is None:

        def _random_color(name: str) -> RGBTuple:
            id, max_label = tissue_label_map[name], num_tissues
            if id <= 0:
                raise ValueError(
                    "Background (label=0) is implicit and not written to file"
                )
            hue = id / (2.0 * max_label) + (id % 2) * 0.5
            hue = min(hue, 1.0)
            return colorsys.hls_to_rgb(hue, 0.5, 1.0)

        tissue_color_map = _random_color

    with open(tissue_list_file_name, "w") as f:
        print("V7", file=f)
        print(f"N{num_tissues}", file=f)
        for label in range(1, num_tissues + 1):
            name = label_tissue_map[label]
            r, g, b = tissue_color_map(name)
            print(
                f"C{r:.2f} {g:.2f} {b:.2f} {0.5:.2f} {name}",
                file=f,
            )


def load_tissue_list(file_name: Path) -> Dict[str, int]:
    """Load tissue list in iSEG format

    Example file:
        V7
        N3
        C0.00 0.00 1.00 0.50 Bone
        C0.00 1.00 0.00 0.50 Fat
        C1.00 0.00 0.00 0.50 Skin
    """
    tissue_label_map = {"Background": 0}
    next_id = 1
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith("C"):
                tissue = line.strip().rsplit(" ", 1)[-1].rstrip()
                if tissue in tissue_label_map:
                    raise KeyError(f"duplicate label '{tissue}' found in '{file_name}'")
                tissue_label_map[tissue] = next_id
                next_id += 1
    return tissue_label_map


def load_tissue_colors(file_name: Path) -> Dict[int, RGBTuple]:
    """Load tissue colors from iSEG format tissue list

    Example file:
        V7
        N3
        C0.00 0.00 1.00 0.50 Bone
        C0.00 1.00 0.00 0.50 Fat
        C1.00 0.00 0.00 0.50 Skin
    """
    tissue_idx = 0
    tissue_color_map = {tissue_idx: (0.0, 0.0, 0.0)}
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith("C"):
                rgb = [float(v.strip()) for v in line.lstrip("C").split(" ")[:3]]
                tissue_idx += 1
                tissue_color_map[tissue_idx] = (rgb[0], rgb[1], rgb[2])
    return tissue_color_map
