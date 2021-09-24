from pathlib import Path

import pytest
from segmantic.prepro import labels


@pytest.fixture
def tissue_map() -> dict:
    """maps tissue -> index"""
    return {"BG": 0, "Bone": 1, "Fat": 2, "Skin": 3}


@pytest.fixture
def tissue2display(tissue_map: dict) -> dict:
    """maps tissue -> human readable label"""
    name_map = {key: key.capitalize() for key in tissue_map}
    name_map["BG"] = "Background"
    return name_map


def test_tissue_list_io(temp_path: Path, tissue_map: dict):
    file_path = temp_path / "tissue.txt"

    labels.save_tissue_list(tissue_map, file_path)

    tissue_dict2 = labels.load_tissue_list(file_path)

    assert len(tissue_map) == len(tissue_dict2)

    tissue_map.pop("BG")
    assert tissue_map == {k: v for k, v in tissue_dict2.items() if k != "BG"}


def test_build_tissue_map(tissue_map: dict, tissue2display: dict):

    omap, i2o = labels.build_tissue_mapping(tissue_map, map)
    assert len(omap) == 3

    for n1 in tissue_map:
        n2 = tissue2display.get(n1, "Other")
        i1 = tissue_map[n1]
        i2 = omap[n2]
        assert i2o[i1] == i2
