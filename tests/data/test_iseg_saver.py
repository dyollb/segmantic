from pathlib import Path
from typing import Dict

import pytest
import SimpleITK as sitk
from monai.transforms import LoadImaged

from segmantic.data.transforms import LabelInfo, iSegSaver


@pytest.fixture
def labels() -> Dict[int, LabelInfo]:
    return {0: ("BG", 0.0, 0.0, 0.0), 1: ("FG", 1.0, 1.0, 1.0)}


def test_iSegSaver(
    tmp_path: Path, labelfield: sitk.Image, labels: Dict[int, LabelInfo]
):
    image_file = tmp_path / "image.nii.gz"
    label_file = tmp_path / "label.nii.gz"
    output_dir = tmp_path / "output"

    sitk.WriteImage(labelfield, label_file)
    sitk.WriteImage(labelfield, image_file)

    dataset = {"image": str(image_file), "label": str(label_file)}
    data = LoadImaged(keys=["image", "label"])(dataset)
    assert "image" in data

    saver = iSegSaver(
        keys=["image", "label"],
        image_key="image",
        label_key="label",
        label_dict=labels,
        output_dir=str(output_dir),
        separate_folder=False,
        allow_missing_keys=True,
    )
    # save with full data
    saver(data)

    output_files = list(output_dir.glob("*.h5"))
    assert len(output_files) == 1
    output_files[0].unlink()

    # save with missing image
    saver({k: data[k] for k in data if "image" not in k})

    output_files = list(output_dir.glob("*.h5"))
    assert len(output_files) == 1
    output_files[0].unlink()

    # save with missing label
    saver({k: data[k] for k in data if "label" not in k})

    output_files = list(output_dir.glob("*.h5"))
    assert len(output_files) == 1
    output_files[0].unlink()
