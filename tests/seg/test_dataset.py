import json
from pathlib import Path
from typing import Tuple

from segmantic.seg import dataset
from segmantic.utils.file_iterators import find_matching_files


def dataset_mockup(
    root_path: Path, label_suffix: str = "", size: int = 3
) -> Tuple[Path, Path]:
    image_dir, labels_dir = root_path / "image", root_path / "label"
    image_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

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
    val_paths = tmp_path.joinpath("val_paths")
    val_paths.mkdir(exist_ok=True)
    image_dir, labels_dir = dataset_mockup(root_path=tmp_path, size=2)
    image_dir_v, labels_dir_v = dataset_mockup(root_path=val_paths, size=1)

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "training": [
                    {
                        "image": f"{image_dir.relative_to(tmp_path) / '*.nii.gz'}",
                        "label": f"{labels_dir.relative_to(tmp_path)/ '*.nii.gz'}",
                    }
                ],
                "validation": [
                    {
                        "image": f"{image_dir_v.relative_to(tmp_path)/'*.nii.gz'}",
                        "label": f"{labels_dir_v.relative_to(tmp_path)/'*.nii.gz'}",
                    }
                ],
            }
        )
    )
    ds = dataset.PairedDataSet.load_from_json(dataset_file)
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()

    # now dump and try to re-load
    dataset_file2 = tmp_path / "dataset_dump.json"
    dataset_file2.write_text(ds.dump_dataset())
    ds = dataset.PairedDataSet.load_from_json(dataset_file2)
    assert len(ds.training_files()) == 2
    assert len(ds.validation_files()) == 1
    ds.check_matching_filenames()


def test_kfold_crossval(tmp_path: Path):
    output_dir = tmp_path / "k_fold_outputs"
    output_dir.mkdir(exist_ok=True)

    datafolds_dir = output_dir / "datafolds"

    image_dir, labels_dir = dataset_mockup(root_path=tmp_path, size=21)
    data_dicts = dataset.PairedDataSet.create_data_dict(
        image_dir=image_dir, labels_dir=labels_dir
    )

    image_dir, labels_dir = dataset_mockup(root_path=tmp_path, size=3)
    test_data_dicts = dataset.PairedDataSet.create_data_dict(
        image_dir=image_dir, labels_dir=labels_dir
    )

    all_datafold_paths = dataset.PairedDataSet.kfold_crossval(
        num_splits=7,
        data_dicts=data_dicts,
        output_dir=datafolds_dir,
        test_data_dicts=test_data_dicts,
    )
    assert len(all_datafold_paths) == 7
    assert len(list(datafolds_dir.glob("*.json"))) == 7

    for dpath in all_datafold_paths:
        assert Path(dpath).is_file()
        temp_ds = dataset.PairedDataSet.load_from_json(dpath)
        temp_ds.check_matching_filenames()


def test_find_matching_files(tmp_path: Path):
    image_dir, labels_dir = dataset_mockup(
        root_path=tmp_path, size=3, label_suffix="_seg"
    )

    tuple1 = find_matching_files([image_dir / "**" / "*.nii.gz"])
    assert len(tuple1) == 3

    tuple2 = find_matching_files(
        [image_dir / "**/*.nii.gz", labels_dir / "**/*_seg.nii.gz"]
    )
    assert len(tuple2) == 3

    tuple2_bad = find_matching_files(
        [image_dir / "**/*.nii.gz", labels_dir / "**/*.nii.gz"]
    )
    assert len(tuple2_bad) == 0
