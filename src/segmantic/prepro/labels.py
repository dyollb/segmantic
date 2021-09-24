import numpy as np
import colorsys
from pathlib import Path
from typing import Callable, Dict, List, Tuple
RGBTuple = Tuple[float, float, float]


def build_tissue_mapping(input_label_map: Dict[str, int], mapper: Callable[[str], str]):
    """build mapping to relable

    Args:
        input_label_map (Dict[str, int]): list of tissue names, e.g. loaded with 'load_tissue_list'
        mapper (Callable[[str], str]): function that maps a tissue name to a new name

    Returns:
        Dict[str, int]: tissue label dict after mapping
        np.ndarray: input-to-output lookup table
    """
    output_label_names = list(sorted(set([mapper(n) for n in input_label_map.keys()])))
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
):
    """save tissue list in iSEG format

    Example:
    from segmantic.prepro import labels
    labels.save_tissue_list({ 'Bone': 1, 'Fat': 2, 'Skin': 3 }, 'tissues.txt')
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

        def random_color(l, max_label) -> RGBTuple:
            if l == 0:
                return (0, 0, 0)
            hue = l / (2.0 * max_label) + (l % 2) * 0.5
            hue = min(hue, 1.0)
            return colorsys.hls_to_rgb(hue, 0.5, 1.0)

        tissue_color_map = lambda n: random_color(tissue_label_map[n], num_tissues)

    with open(tissue_list_file_name, "w") as f:
        print("V7", file=f)
        print("N%d" % num_tissues, file=f)
        for label in range(1, num_tissues + 1):
            name = label_tissue_map[label]
            r, g, b = tissue_color_map(name)
            print(
                "C%.2f %.2f %.2f %.2f %s" % (r, g, b, 0.5, name),
                file=f,
            )


def load_tissue_list(file_name: Path) -> Dict[str, int]:
    """load tissue list in iSEG format

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
                tissue = line.rsplit(" ", 1)[-1].rstrip()
                if tissue in tissue_label_map:
                    raise KeyError(
                        "duplicate label '%s' found in '%s'" % (tissue, file_name)
                    )
                tissue_label_map[tissue] = next_id
                next_id += 1
    return tissue_label_map
