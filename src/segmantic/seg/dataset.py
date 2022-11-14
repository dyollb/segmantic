import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
from sklearn.model_selection import KFold

from ..utils.json import PathEncoder


def find_matching_files(input_globs: List[Path], verbose: bool = True):
    dir_0 = Path(input_globs[0].anchor)
    glob_0 = str(input_globs[0].relative_to(dir_0))
    ext_0 = input_globs[0].name.rsplit("*")[-1]

    candidate_files = {p.name.replace(ext_0, ""): [p] for p in dir_0.glob(glob_0)}

    for other_glob in input_globs[1:]:
        dir_i = Path(other_glob.anchor)
        glob_i = str(other_glob.relative_to(dir_i))
        ext_i = other_glob.name.rsplit("*")[-1]

        for p in dir_i.glob(glob_i):
            key = p.name.replace(ext_i, "")
            if key in candidate_files:
                candidate_files[key].append(p)
            elif verbose:
                print(f"No match found for {key} : {p}")

    output_files = [v for v in candidate_files.values() if len(v) == len(input_globs)]

    if verbose:
        print(f"Number of files in {input_globs[0]}: {len(candidate_files)}")
        print(f"Number of tuples: {len(output_files)}\n")

    return output_files


def create_data_dict(
    list_to_convert: List[Dict[str, str]], p: Path, data_dicts: List[Dict[str, Path]]
):
    for element in list_to_convert:
        # special case: absolute paths
        if Path(element["image"]).is_absolute():
            image_files = [Path(element["image"])]
            label_files = [Path(element["label"])]
        else:
            image_files = list(p.parent.glob(element["image"]))
            label_files = list(p.parent.glob(element["label"]))
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
        image_dir: Path = Path(),
        image_glob: str = "*.nii.gz",
        labels_dir: Path = Path(),
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

    def training_files(self) -> Sequence[Dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for training"""
        return self._train_files

    def validation_files(self) -> Sequence[Dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for validation"""
        return self._val_files

    def test_files(self) -> Sequence[Dict[str, Path]]:
        """Get list of 'image'/'label' pairs (dictionary) for test"""
        return self._test_files

    def _create_split(
        self,
        data_dicts: List[Dict[str, Path]],
        valid_split: float,
        shuffle: bool,
        random_seed: int = None,
        max_files: int = 0,
        test_data_dicts: List[Dict[str, Path]] = [],
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
                "test": self._test_files,
            },
            cls=PathEncoder,
        )

    @staticmethod
    def create_data_dict(
        image_dir: Path = Path(),
        image_glob: str = "*.nii.gz",
        labels_dir: Path = Path(),
        labels_glob: str = "*.nii.gz",
    ) -> List[Dict[str, Path]]:

        assert image_dir.is_dir() and labels_dir.is_dir()

        if Path(image_glob).is_absolute():
            image_glob = str(Path(image_glob).relative_to(image_dir))
        if Path(labels_glob).is_absolute():
            labels_glob = str(Path(labels_glob).relative_to(labels_dir))

        image_files = list(image_dir.glob(image_glob))
        label_files = list(labels_dir.glob(labels_glob))
        assert len(image_files) == len(label_files)

        data_dicts: List[Dict[str, Path]] = []
        for i, o in zip(sorted(image_files), sorted(label_files)):
            data_dicts.append({"image": i, "label": o})
        return data_dicts

    @staticmethod
    def kfold_crossval(
        num_splits: int,
        data_dicts: List[Dict[str, Path]],
        output_dir: Path,
        test_data_dicts: List[Dict[str, Path]] = [],
    ) -> List:
        kf = KFold(n_splits=num_splits)

        all_fold_files_dir = output_dir.joinpath("datafolds")
        all_fold_files_dir.mkdir(exist_ok=True)

        image_idx = np.arange(len(data_dicts))
        all_dataset_paths: List[Path] = []

        for count, (train_idx, val_idx) in enumerate(kf.split(image_idx, image_idx)):
            temp_dataset = PairedDataSet()
            temp_dataset._train_files = [data_dicts[i] for i in train_idx]
            temp_dataset._val_files = [data_dicts[i] for i in val_idx]
            temp_dataset._test_files = test_data_dicts

            temp_dataset_path = all_fold_files_dir.joinpath(f"fold_{count}.json")
            temp_dataset_path.write_text(temp_dataset.dump_dataset())
            all_dataset_paths.append(temp_dataset_path)

        return all_dataset_paths

    @staticmethod
    def load_from_json(
        file_path: Union[Path, List[Path]],
    ):
        """Loads one or more datasets from json descriptor files and returns a single combined dataset

        The json file convention follows the MSD dataset, also used e.g. by nnUNet.

        The training data is loaded from the 'training' section. Glob expressions are
        supported as well as a full list of files:
        {
            "training": [{"image": "image/*.nii.gz", "label": "label/*.nii.gz"}],
        }
        """

        if isinstance(file_path, (Path, str)):
            file_path = [file_path]

        data_dicts_train: List[Dict[str, Path]] = []
        data_dicts_val: List[Dict[str, Path]] = []
        data_dicts_test: List[Dict[str, Path]] = []

        for p in [Path(f) for f in file_path]:
            ds = json.loads(p.read_text())
            training = ds["training"]
            validation = ds["validation"]
            test = ds["test"] if "test" in ds else []

            data_dicts_train = create_data_dict(
                list_to_convert=training, p=p, data_dicts=data_dicts_train
            )

            data_dicts_val = create_data_dict(
                list_to_convert=validation, p=p, data_dicts=data_dicts_val
            )

            data_dicts_test = create_data_dict(
                list_to_convert=test, p=p, data_dicts=data_dicts_test
            )

        combined_ds = PairedDataSet()
        combined_ds._train_files = data_dicts_train
        combined_ds._val_files = data_dicts_val
        if test is not None:
            combined_ds._test_files = data_dicts_test
        return combined_ds
