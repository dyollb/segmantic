import json

from pathlib import Path
from typing import Tuple

from segmantic.seg import dataset


def dataset_mockup(root_path: Path, size: int = 3) -> Tuple[Path, Path]:
    image_dir, label_dir = root_path / "image", root_path / "label"
    image_dir.mkdir()
    label_dir.mkdir()

    for idx in range(size):
        (image_dir / f"img-{idx}.nii.gz").touch()
        (label_dir / f"img-{idx}.nii.gz").touch()
    return image_dir, label_dir


def test_PairedDataSet(tmp_path: Path):
    image_dir, label_dir = dataset_mockup(root_path=tmp_path, size=3)

    ds = dataset.PairedDataSet(
        image_dir=image_dir, label_dir=label_dir, valid_split=0.2
    )
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()

    ds = dataset.PairedDataSet(image_dir=image_dir, label_dir=label_dir, valid_split=0)
    assert len(ds.training_files()) == 3
    assert len(ds.validation_files()) == 0
    ds.check_matching_filenames()


def test_load_from_json(tmp_path: Path):
    image_dir, label_dir = dataset_mockup(root_path=tmp_path, size=3)

    (tmp_path / "dataset.json").write_text(
        json.dumps({"image_dir": f"{image_dir}", "label_dir": f"{label_dir}"})
    )

    ds = dataset.PairedDataSet.load_from_json(
        tmp_path / "dataset.json", valid_split=0.2
    )
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()
