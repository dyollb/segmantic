import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.model_selection import KFold

from ..util.json import PathEncoder


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


class PairedDataSet(object):
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
        cross_val: bool = False,
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

    def _create_split(
        self,
        data_dicts: List[Dict[str, Path]],
        valid_split: Optional[float] = None,
        shuffle: Optional[bool] = None,
        random_seed: Optional[int] = None,
        max_files: Optional[int] = 0,
        cross_val: bool = False,
        train_idx: Optional[List[int]] = None,
        val_idx: Optional[List[int]] = None,
    ):
        if not cross_val:
            assert shuffle is not None and random_seed is not None
            if shuffle:
                my_random = random.Random(random_seed)
                my_random.shuffle(data_dicts)

            assert max_files is not None
            num_total = len(data_dicts)
            if max_files > 0:
                num_total = min(num_total, max_files)

            assert valid_split is not None
            num_valid = int(valid_split * num_total)
            if num_total > 1 and valid_split > 0:
                num_valid = max(num_valid, 1)

            # use first `num_valid` files for validation, rest for training
            self._train_files = data_dicts[num_valid:num_total]
            self._val_files = data_dicts[:num_valid]
        else:
            if train_idx is not None and val_idx is not None:
                self._train_files = [data_dicts[index] for index in train_idx]
                self._val_files = [data_dicts[index] for index in val_idx]
            else:
                raise ValueError("Please Assign train and val_idx")

    def check_matching_filenames(self):
        """Check if the files names are identical except for any prefix or suffix"""
        for d in self._train_files + self._val_files:
            input_stem = d["image"].stem.replace(".nii", "").lower()
            output_stem = d["label"].stem.replace(".nii", "").lower()
            if not ((input_stem in output_stem) or (output_stem in input_stem)):
                raise RuntimeError(
                    f"The pair image/label pair {d['image']} : {d['label']} doesn't correspond."
                )

    def dump_dataset(self, cross_val: bool = False) -> str:
        if not cross_val:
            return json.dumps(
                {
                    "training": self._train_files + self._val_files,
                    "split": [len(self._train_files), len(self._val_files)],
                },
                cls=PathEncoder,
            )
        else:
            return json.dumps(
                {
                    "training": self._train_files,
                    "validation": self._val_files,
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
        num_splits: int, data_dicts: List[Dict[str, Path]], output_dir: Path
    ) -> List:
        kf = KFold(n_splits=num_splits)

        all_fold_files_dir = output_dir.joinpath("datafolds")
        all_fold_files_dir.mkdir(exist_ok=True)

        image_idx = np.arange(len(data_dicts))
        all_dataset_paths: List[Path] = []

        for count, (train_idx, test_idx) in enumerate(kf.split(image_idx, image_idx)):
            temp_dataset = PairedDataSet()
            temp_dataset._create_split(
                data_dicts=data_dicts,
                cross_val=True,
                train_idx=train_idx,
                val_idx=test_idx,
            )

            temp_dataset_path = all_fold_files_dir.joinpath(f"fold_{count}")
            temp_dataset_path.write_text(temp_dataset.dump_dataset(cross_val=True))
            all_dataset_paths.append(temp_dataset_path)

        return all_dataset_paths

    @staticmethod
    def load_from_json(
        file_path: Union[Path, List[Path]],
        *,
        valid_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = None,
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

        data_dicts: List[Dict[str, Path]] = []

        for p in (Path(f) for f in file_path):
            training = json.loads(p.read_text())["training"]
            for d in training:
                # special case: absolute paths
                if Path(d["image"]).is_absolute():
                    image_files = [Path(d["image"])]
                    label_files = [Path(d["label"])]
                else:
                    image_files = list(p.parent.glob(d["image"]))
                    label_files = list(p.parent.glob(d["label"]))
                    assert len(image_files) == len(label_files)

                for i, o in zip(sorted(image_files), sorted(label_files)):
                    data_dicts.append({"image": i, "label": o})

        combined_ds = PairedDataSet()
        combined_ds._create_split(data_dicts, valid_split, shuffle, random_seed)
        return combined_ds

    @staticmethod
    def load_from_cross_val_json(
        file_path: Path,
    ):
        """Loads one or more datasets from json descriptor files and returns a single combined dataset

        The json file convention follows the MSD dataset, also used e.g. by nnUNet.

        The training data is loaded from the 'training' section. Glob expressions are
        supported as well as a full list of files:
        {
            "training": [{"image": "image/*.nii.gz", "label": "label/*.nii.gz"}],
        }
        """

        data_dicts_train: List[Dict[str, Path]] = []
        data_dicts_val: List[Dict[str, Path]] = []

        training = json.loads(file_path.read_text())["training"]
        validation = json.loads(file_path.read_text())["validation"]

        for t, v in zip(training, validation):
            # special case: absolute paths
            if Path(t["image"]).is_absolute():
                image_files_t = [Path(t["image"])]
                label_files_t = [Path(t["label"])]
                image_files_v = [Path(v["image"])]
                label_files_v = [Path(v["label"])]
            else:
                image_files_t = list(file_path.parent.glob(t["image"]))
                label_files_t = list(file_path.parent.glob(t["label"]))
                image_files_v = list(file_path.parent.glob(v["image"]))
                label_files_v = list(file_path.parent.glob(v["label"]))
            assert len(image_files_t) == len(label_files_t)
            assert len(image_files_v) == len(label_files_v)

            for i_t, o_t, i_v, o_v in zip(
                sorted(image_files_t),
                sorted(label_files_t),
                sorted(label_files_v),
                sorted(label_files_v),
            ):
                data_dicts_train.append({"image": i_t, "label": o_t})
                data_dicts_val.append({"image": i_v, "label": o_v})

        combined_ds = PairedDataSet()
        combined_ds._train_files = data_dicts_train
        combined_ds._val_files = data_dicts_val
        return combined_ds
