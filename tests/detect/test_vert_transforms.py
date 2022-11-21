import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import SimpleITK as sitk
from monai.transforms import AsDiscreted, Compose, EnsureChannelFirstd, LoadImaged
from numpy.testing import assert_almost_equal

from segmantic.detect.transforms import (
    BoundingBoxd,
    EmbedVert,
    ExtractVertPosition,
    LoadVert,
    SaveVert,
)
from segmantic.image.processing import make_image

KEY_0 = "point_0"
KEY_1 = "point_1"
TEST_POINT_0 = [4.95, -5.0, 3.5]
TEST_POINT_1 = [13.45, -29.0, 3.5]


@pytest.fixture
def landmarks() -> Dict[str, List[float]]:
    data = {
        KEY_0: TEST_POINT_0,
        KEY_1: TEST_POINT_1,
    }
    return data


@pytest.fixture
def example_image() -> sitk.Image:
    image = make_image(shape=[30, 20, 10], spacing=[1.0, 0.85, 2.5], value=0)
    mat = np.eye(3, 3)
    mat[:2, :] = 0
    mat[0, 1] = -1.0
    mat[1, 0] = 1.0
    image.SetDirection(mat.flatten().tolist())
    image.SetOrigin([1] * 3)
    return image


def test_LoadVert(tmp_path: Path, landmarks: Dict[str, List[float]]):
    tmp_file = tmp_path / "points.json"
    tmp_file.write_text(json.dumps(landmarks))

    data = LoadVert(keys="vert", meta_key_postfix="meta")({"vert": tmp_file})
    assert "vert" in data
    verts = data["vert"]
    assert isinstance(verts, dict)
    assert all(id in verts for id in (1, 2))
    assert_almost_equal(verts[1], np.array(TEST_POINT_0))
    assert_almost_equal(verts[2], np.array(TEST_POINT_1))

    meta_key = "vert_meta"
    assert meta_key in data
    assert data[meta_key]["id_map"][KEY_0] == 1
    assert data[meta_key]["id_map"][KEY_1] == 2


def test_EmbedVert(
    tmp_path: Path, landmarks: Dict[str, List[float]], example_image: sitk.Image
):
    vert_file = tmp_path / "points.json"
    vert_file.write_text(json.dumps(landmarks))

    img_file = tmp_path / "image.nii.gz"
    sitk.WriteImage(example_image, img_file)

    tr = Compose(
        [
            LoadImaged(keys="image"),
            LoadVert(keys="vert"),
            EmbedVert(keys="vert", ref_key="image"),
        ]
    )
    data = tr({"vert": vert_file, "image": img_file})
    assert "vert" in data
    assert "image" in data

    img = data["vert"]
    assert_almost_equal(np.min(img), 0.0)
    assert_almost_equal(np.max(img), len(landmarks))


def test_Vert_RoundTrip(
    tmp_path: Path, landmarks: Dict[str, List[float]], example_image: sitk.Image
):
    vert_file = tmp_path / "points.json"
    vert_file.write_text(json.dumps(landmarks))

    img_file = tmp_path / "image.nii.gz"
    sitk.WriteImage(example_image, img_file)

    tr = Compose(
        [
            LoadImaged(keys="image"),
            LoadVert(keys="vert"),
            EmbedVert(keys="vert", ref_key="image"),
            EnsureChannelFirstd(keys="vert"),
            AsDiscreted(keys="vert", to_onehot=3),
            ExtractVertPosition(keys="vert", threshold=0.5),
            SaveVert(keys="vert", output_dir=str(tmp_path)),
        ]
    )
    tr({"vert": vert_file, "image": img_file})

    output_file = tmp_path / "points" / "points_trans.json"
    assert output_file.exists()
    verts = json.loads(output_file.read_text())
    assert len(verts) == len(landmarks)
    assert all(k in verts for k in landmarks)

    assert_almost_equal(np.array(verts[KEY_0]), np.array(TEST_POINT_0), decimal=4)
    assert_almost_equal(np.array(verts[KEY_1]), np.array(TEST_POINT_1), decimal=4)


def test_BoundingBox(tmp_path: Path, example_image: sitk.Image):
    img_file = tmp_path / "image.nii.gz"
    sitk.WriteImage(example_image, img_file)

    tr = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            BoundingBoxd(keys="image", result="result", bbox="bbox"),
        ]
    )
    d = tr({"image": img_file})
    assert "result" in d
    assert "bbox" in d["result"]
    bbox: list = d["result"]["bbox"]
    assert bbox[0] == bbox[1]

    arr = sitk.GetArrayFromImage(example_image)
    arr[5:7, :, 10:23] = 1
    image2 = sitk.GetImageFromArray(arr)
    sitk.WriteImage(image2, img_file)
    d = tr({"image": img_file})
    assert "result" in d
    assert "bbox" in d["result"]
    bbox = d["result"]["bbox"]
    print(bbox)
    np.testing.assert_array_equal(bbox, np.asarray([[10, 0, 5], [23, 20, 7]]))
