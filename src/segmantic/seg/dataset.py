import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Union

from ..util.json import PathEncoder


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
    ):
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
        valid_split: float,
        shuffle: bool,
        random_seed: int = None,
        max_files: int = 0,
    ):

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
                "training": self._train_files + self._val_files,
                "split": [len(self._train_files), len(self._val_files)],
            },
            cls=PathEncoder,
        )

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
