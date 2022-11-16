from pathlib import Path

import pytest

from segmantic.image import labels


@pytest.fixture
def tissue_map() -> dict:
    """maps tissue -> index"""
    return {"Background": 0, "Bone": 1, "Fat": 2, "Skin": 3}


def test_tissue_list_io(tmp_path: Path, tissue_map: dict) -> None:
    file_path = tmp_path / "tissue.txt"

    labels.save_tissue_list(tissue_map, file_path)

    tissue_dict2 = labels.load_tissue_list(file_path)

    assert len(tissue_map) == len(tissue_dict2)

    assert tissue_map == tissue_dict2

    tissue_colors = labels.load_tissue_colors(file_path)

    assert len(tissue_colors) == len(tissue_dict2)


def test_build_tissue_map(tissue_map: dict) -> None:

    # map "Skin" and "Fat" to "Other_tissue"
    def _map_name(name):
        if name == "Background":
            return name
        if name == "Bone":
            return "Bone"
        return "Other_tissue"

    omap, i2o = labels.build_tissue_mapping(tissue_map, _map_name)

    assert len(omap) == 3

    assert omap == {_map_name(n1): i2o[i1] for n1, i1 in tissue_map.items()}
