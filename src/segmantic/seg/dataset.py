import random
from pathlib import Path
from typing import Dict, List, Sequence


class PairedDataSet(object):
    def __init__(
        self,
        input_dir: Path = Path(""),
        input_key: str = "image",
        output_dir: Path = Path(""),
        output_key: str = "label",
        file_extension="*.nii.gz",
        valid_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = None,
        max_files: int = 0,
    ):
        super().__init__()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        common_names: List[str] = []
        output_files = [p.name for p in output_dir.glob(file_extension)]
        for f in (p.name for p in input_dir.iterdir()):
            if f in output_files:
                common_names.append(f)

        if shuffle:
            my_random = random.Random(random_seed)
            my_random.shuffle(common_names)

        data_dicts = [
            {input_key: input_dir / f, output_key: output_dir / f} for f in common_names
        ]
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
