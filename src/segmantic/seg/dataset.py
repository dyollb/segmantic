from typing import Sequence, Dict
from pathlib import Path


class DataSet(object):
    def __init__(
        self,
        image_dir: Path = Path(""),
        labels_dir: Path = Path(""),
        valid_split: float = 0.2,
    ):
        super().__init__()

        image_files = sorted(Path(image_dir).glob("*.nii.gz"))
        label_files = sorted(Path(labels_dir).glob("*.nii.gz"))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(image_files, label_files)
        ]
        num_total = len(data_dicts)
        num_valid = int(valid_split * num_total)

        # use first `N` files for validation, rest for training
        self._train_files, self._val_files = (
            data_dicts[num_valid:],
            data_dicts[:num_valid],
        )

    def training_files(self) -> Sequence[Dict[str, Path]]:
        return self._train_files

    def validation_files(self) -> Sequence[Dict[str, Path]]:
        return self._val_files
