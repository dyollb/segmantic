from pathlib import Path

import pytest
from segmantic.prepro import labels


@pytest.fixture
def tissue_map() -> dict:
    """maps tissue -> index"""
    return {"Background": 0, "Bone": 1, "Fat": 2, "Skin": 3}


def test_tissue_list_io(tmp_path: Path, tissue_map: dict):
    file_path = tmp_path / "tissue.txt"

    labels.save_tissue_list(tissue_map, file_path)

    tissue_dict2 = labels.load_tissue_list(file_path)

    assert len(tissue_map) == len(tissue_dict2)

    assert tissue_map == tissue_dict2


def test_build_tissue_map(tissue_map: dict):

    # map "Skin" and "Fat" to "Other_tissue"
    def map(name):
        if name == "Background":
            return name
        if name == "Bone":
            return "Bone"
        return "Other_tissue"

    omap, i2o = labels.build_tissue_mapping(tissue_map, map)
    assert len(omap) == 3

    for n1 in tissue_map:
        n2 = map(n1)
        i1 = tissue_map[n1]
        i2 = omap[n2]
        assert i2o[i1] == i2
