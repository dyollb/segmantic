import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.model_selection import KFold

from ..utils.file_iterators import find_matching_files
from ..utils.json import PathEncoder


def create_data_dict(
    list_to_convert: list[dict[str, str]],
    data_dir: Path,
    data_dicts: list[dict[str, Path]],
) -> list[dict[str, Path]]:
    """Handle glob expressions to build datalist"""

    for element in list_to_convert:
        # special case: absolute paths
        if Path(element["image"]).is_absolute():
            image_files = [Path(element["image"])]
            label_files = [Path(element["label"])]
        else:
            image_files = list(data_dir.glob(element["image"]))
            label_files = list(data_dir.glob(element["label"]))
        assert len(image_files) == len(label_files)

        for i_element, o_element in zip(
            sorted(image_files),
            sorted(label_files),
        ):
            data_dicts.append({"image": i_element, "label": o_element})

    return data_dicts


class PairedDataSet:
    def __init__(
        self,
        image_dir: Optional[Path] = None,
        image_glob: str = "*.nii.gz",
        labels_dir: Optional[Path] = None,
        labels_glob: str = "*.nii.gz",
        *,
        valid_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = None,
        max_files: int = 0,
    ):
        data_dicts = self.create_data_dict(
            image_dir, image_glob, labels_dir, labels_glob
        )
        self._create_split(data_dicts, valid_split, shuffle, random_seed, max_files)

    def training_files(self) -> Sequence[dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for training"""
        return self._train_files

    def validation_files(self) -> Sequence[dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for validation"""
        return self._val_files

    def test_files(self) -> Sequence[dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for test"""
        return self._test_files

    def _create_split(
        self,
        data_dicts: list[dict[str, Path]],
        valid_split: float,
        shuffle: bool,
        random_seed: int = None,
        max_files: int = 0,
        test_data_dicts: list[dict[str, Path]] = [],
    ):
        self._test_files = test_data_dicts

        if shuffle:
            my_random = random.Random(random_seed)
            my_random.shuffle(data_dicts)

        num_total = len(data_dicts)
        if max_files > 0:
            num_total = min(num_total, max_files)

        num_valid = int(valid_split * num_total)
        if num_total > 1 and valid_split > 0:
            num_valid = max(num_valid, 1)

        # use first `num_valid` files for validation, rest for training
        self._train_files = data_dicts[num_valid:num_total]
        self._val_files = data_dicts[:num_valid]

    def check_matching_filenames(self):
        """Check if the files names are identical except for any prefix or suffix"""
        for d in self._train_files + self._val_files:
            input_stem = d["image"].stem.replace(".nii", "").lower()
            output_stem = d["label"].stem.replace(".nii", "").lower()
            if not ((input_stem in output_stem) or (output_stem in input_stem)):
                raise RuntimeError(
                    f"The pair image/label pair {d['image']} : {d['label']} doesn't correspond."
                )

    def dump_dataset(self) -> str:
        return json.dumps(
            {
                "training": self._train_files,
                "validation": self._val_files,
                "test": [t["image"] for t in self._test_files],
            },
            cls=PathEncoder,
        )

    @staticmethod
    def create_data_dict(
        image_dir: Optional[Path] = None,
        image_glob: str = "*.nii.gz",
        labels_dir: Optional[Path] = None,
        labels_glob: str = "*.nii.gz",
    ) -> list[dict[str, Path]]:

        data_dicts: list[dict[str, Path]] = []
        if image_dir is None or labels_dir is None:
            return data_dicts

        assert image_dir.is_dir() and labels_dir.is_dir()
        if Path(image_glob).is_absolute():
            image_glob = str(Path(image_glob).relative_to(image_dir))
        if Path(labels_glob).is_absolute():
            labels_glob = str(Path(labels_glob).relative_to(labels_dir))

        matches = find_matching_files(
            [image_dir / image_glob, labels_dir / labels_glob]
        )

        for p in matches:
            data_dicts.append({"image": p[0], "label": p[1]})
        return data_dicts

    @staticmethod
    def kfold_crossval(
        num_splits: int,
        data_dicts: list[dict[str, Path]],
        output_dir: Path,
        test_data_dicts: list[dict[str, Path]] = [],
        shuffle: bool = True,
        random_seed: int = None,
    ) -> list:
        kf = KFold(n_splits=num_splits)

        if shuffle:
            my_random = random.Random(random_seed)
            my_random.shuffle(data_dicts)

        output_dir.mkdir(exist_ok=True, parents=True)

        image_idx = np.arange(len(data_dicts))
        all_dataset_paths: list[Path] = []

        for count, (train_idx, val_idx) in enumerate(kf.split(image_idx, image_idx)):
            dataset = PairedDataSet()
            dataset._train_files = [data_dicts[i] for i in train_idx]
            dataset._val_files = [data_dicts[i] for i in val_idx]
            dataset._test_files = test_data_dicts

            dataset_path = output_dir / f"fold_{count}.json"
            dataset_path.write_text(dataset.dump_dataset())
            all_dataset_paths.append(dataset_path)

        return all_dataset_paths

    @staticmethod
    def load_from_json(
        datalist_paths: Union[Path, list[Path]],
    ):
        """Loads one or more datasets from json descriptor files and returns a single combined dataset

        The json file convention follows the MSD dataset, also used e.g. by nnUNet.

        The training data is loaded from the 'training' section. Glob expressions are
        supported as well as a full list of files:
        {
            "training": [{"image": "image/*.nii.gz", "label": "label/*.nii.gz"}],
        }
        """

        if isinstance(datalist_paths, (Path, str)):
            datalist_paths = [datalist_paths]

        data_dicts_train: list[dict[str, Path]] = []
        data_dicts_val: list[dict[str, Path]] = []
        data_dicts_test: list[dict[str, Path]] = []

        for json_path in [Path(f) for f in datalist_paths]:
            ds: dict = json.loads(json_path.read_text())
            training = ds["training"]
            validation = ds["validation"]
            test = ds.get("test", [])

            data_dicts_train = create_data_dict(
                list_to_convert=training,
                data_dir=json_path.parent,
                data_dicts=data_dicts_train,
            )

            data_dicts_val = create_data_dict(
                list_to_convert=validation,
                data_dir=json_path.parent,
                data_dicts=data_dicts_val,
            )

            data_dicts_test = [{"image": Path(f)} for f in test]

        combined_ds = PairedDataSet()
        combined_ds._train_files = data_dicts_train
        combined_ds._val_files = data_dicts_val
        if test is not None:
            combined_ds._test_files = data_dicts_test
        return combined_ds
