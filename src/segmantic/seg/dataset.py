from typing import Sequence, Dict
from pathlib import Path


class DataSet(object):
    def __init__(
        self,
        image_dir: Path = Path(""),
        labels_dir: Path = Path(""),
        num_valid: int = 4,
    ):
        super().__init__()

        image_files = sorted(image_dir.glob("*.nii.gz"))
        label_files = sorted(labels_dir.glob("*.nii.gz"))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(image_files, label_files)
        ]
        
        print(len(data_dicts))
        validation_set = int(0.3*len(data_dicts))
        print(validation_set)

        # use first `N` files for validation, rest for training
        self._train_files, self._val_files = (
            data_dicts[validation_set:],
            data_dicts[:validation_set],
        )

    def training_files(self) -> Sequence[Dict[str, Path]]:
        return self._train_files

    def validation_files(self) -> Sequence[Dict[str, Path]]:
        return self._val_files
