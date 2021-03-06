import json
from pathlib import Path
from typing import Tuple

from segmantic.seg import dataset


def dataset_mockup(
    root_path: Path, label_suffix: str = "", size: int = 3
) -> Tuple[Path, Path]:
    image_dir, labels_dir = root_path / "image", root_path / "label"
    image_dir.mkdir()
    labels_dir.mkdir()

    for idx in range(size):
        (image_dir / f"img-{idx}.nii.gz").touch()
        (labels_dir / f"img-{idx}{label_suffix}.nii.gz").touch()
    return image_dir, labels_dir


def test_PairedDataSet(tmp_path: Path):
    image_dir, labels_dir = dataset_mockup(root_path=tmp_path, size=3)

    ds = dataset.PairedDataSet(
        image_dir=image_dir, labels_dir=labels_dir, valid_split=0.2
    )
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()

    ds = dataset.PairedDataSet(
        image_dir=image_dir, labels_dir=labels_dir, valid_split=0
    )
    assert len(ds.training_files()) == 3
    assert len(ds.validation_files()) == 0
    ds.check_matching_filenames()


def test_load_from_json(tmp_path: Path):
    image_dir, labels_dir = dataset_mockup(root_path=tmp_path, size=3)

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "training": [
                    {
                        "image": f"{image_dir.name}/*.nii.gz",
                        "label": f"{labels_dir.name}/*.nii.gz",
                    }
                ]
            }
        )
    )

    ds = dataset.PairedDataSet.load_from_json(dataset_file, valid_split=0.2)
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()

    # now dump and try to re-load
    dataset_file2 = tmp_path / "dataset_dump.json"
    dataset_file2.write_text(ds.dump_dataset())
    ds = dataset.PairedDataSet.load_from_json(dataset_file2, valid_split=0.2)
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()


def test_find_matching_files(tmp_path: Path):
    image_dir, labels_dir = dataset_mockup(
        root_path=tmp_path, size=3, label_suffix="_seg"
    )

    tuple1 = dataset.find_matching_files([image_dir / "**" / "*.nii.gz"])
    assert len(tuple1) == 3

    tuple2 = dataset.find_matching_files(
        [image_dir / "**/*.nii.gz", labels_dir / "**/*_seg.nii.gz"]
    )
    assert len(tuple2) == 3

    tuple2_bad = dataset.find_matching_files(
        [image_dir / "**/*.nii.gz", labels_dir / "**/*.nii.gz"]
    )
    assert len(tuple2_bad) == 0
