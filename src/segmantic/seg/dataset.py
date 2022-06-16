import random
from pathlib import Path
from typing import Dict, List, Sequence


class PairedDataSet(object):
    def __init__(
        self,
        input_dir: Path = Path(""),
        input_key: str = "image",
        input_glob: str = "*.nii.gz",
        output_dir: Path = Path(""),
        output_key: str = "label",
        output_glob: str = "*.nii.gz",
        valid_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = None,
        check_matching_filenames: bool = False,
        max_files: int = 0,
    ):
        super().__init__()

        input_files = list(input_dir.glob(input_glob))
        output_files = list(output_dir.glob(output_glob))
        assert len(input_files) == len(output_files)

        data_dicts = []
        for i, o in zip(sorted(input_files), sorted(output_files)):
            if check_matching_filenames:
                istem = i.stem.replace(".nii", "").lower()
                ostem = o.stem.replace(".nii", "").lower()
                if not ((istem in ostem) or (ostem in istem)):
                    raise RuntimeError(
                        f"The pair image/label pair {i} : {o} doesn't correspond."
                    )
            data_dicts.append({input_key: i, output_key: o})

        if shuffle:
            my_random = random.Random(random_seed)
            my_random.shuffle(data_dicts)

        num_total = (
            len(data_dicts) if max_files == 0 else min(max_files, len(data_dicts))
        )
        num_valid = max(int(valid_split * num_total), 1)

        # use first `num_valid` files for validation, rest for training
        self._train_files, self._val_files = (
            data_dicts[num_valid:num_total],
            data_dicts[:num_valid],
        )

    def training_files(self) -> Sequence[Dict[str, Path]]:
        return self._train_files

    def validation_files(self) -> Sequence[Dict[str, Path]]:
        return self._val_files
