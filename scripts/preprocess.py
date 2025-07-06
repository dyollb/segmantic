from pathlib import Path

import typer
from monai.transforms import (
    Compose,
    CropForegroundd,
    ForegroundMaskd,
    LoadImaged,
    NormalizeIntensityd,
    SaveImaged,
)

from segmantic.seg.transforms import SelectChanneld


def pre_process(input_file: Path, output_dir: Path, margin: int = 0, channel: int = 0):

    transforms = Compose(
        [
            LoadImaged(
                keys="img",
                reader="ITKReader",
                image_only=False,
                ensure_channel_first=True,
            ),
            SelectChanneld(keys="img", channel=channel, new_key_postfix="_0"),
            ForegroundMaskd(keys="img_0", invert=True, new_key_prefix="mask"),
            CropForegroundd(
                keys="img", source_key="maskimg_0", allow_smaller=False, margin=margin
            ),
            NormalizeIntensityd(keys="img"),
            SaveImaged(
                keys="img",
                writer="ITKWriter",
                output_dir=output_dir,
                output_postfix="",
                resample=False,
                separate_folder=False,
                print_log=True,
            ),
        ]
    )
    transforms({"img": input_file})


if __name__ == "__main__":
    typer.run(pre_process)
