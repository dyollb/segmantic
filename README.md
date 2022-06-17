# Semantic segmentation
[![Build Actions Status](https://github.com/dyollb/segmantic/workflows/CI/badge.svg)](https://github.com/dyollb/segmantic/actions)
[![Documentation Status](https://github.com/dyollb/segmantic/workflows/Docs/badge.svg)](https://github.com/dyollb/segmantic/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://https://opensource.org/licenses/MIT)

Semantic segmentation and image-to-image translation based on AI. This repo collects methods for pre-process data, style transfer and semantic segmentation.


## Installation

Dependencies for specific versions of CUDA are listed in `requirements/cu113.txt`, `requirements/cu111.txt`, etc.. It is advisable to install the package in a virtual environment, e.g. using `venv`
```
cd /your/working/directory
python -m venv .venv
```
Activate it using e.g. `source .venv/bin/activate` on Linux/Mac and `.venv\Scripts\activate.bat` on Windows.


To install this repo (this will install all dependencies):
```
pip install git+https://github.com/dyollb/segmantic.git#egg=segmantic
```
Or in edit/dev mode
```
pip install -e git+https://github.com/dyollb/segmantic.git#egg=segmantic[dev]
```

On Windows installing torch with GPU support is slightly more involved. Make sure to first install torch matching the installed CUDA version first or use the requirements files, e.g. for CUDA 11.3
```
--find-links https://download.pytorch.org/whl/cu113/torch_stable.html
torch==1.11.0+cu113
torchvision==0.12.0+cu113
```

## Example scripts

Run training:
```
python src/segmantic/scripts/run_monai_unet.py train --help
python src/segmantic/scripts/run_monai_unet.py train -i work/inputs/images -l work/inputs/labels -t work/inputs/labels.txt -r work/outputs
```

Or with a config file - first create empty config file:
```
python scripts\run_monai_unet.py train-config -c config.json --print-defaults
```

Edit `config.json` e.g. to
```
{
    "image_dir": "work/inputs/images",
    "labels_dir": "work/inputs/labels",
    "tissue_list": "work/inputs/labels.txt",
    "output_dir": "work/outputs",
    "checkpoint_file": null,
    "num_channels": 1,
    "spatial_dims": 3,
    "max_epochs": 500,
    "augment_intensity": true,
    "augment_spatial": false,
    "mixed_precision": true,
    "cache_rate": 1.0,
    "save_nifti": true,
    "gpu_ids": [
        0
    ]
}
```

Now run training:
```
python scripts\run_monai_unet.py train-config -c config.json
```

## What is this tisse_list?
The example above included a tissue_list option. This is a path to a text file specifying the labels contained in a segmented image. By convention the 'label=0' is the background and is ommited from the the format. A segmentation with three tissues 'Bone'=1, 'Fat'=2, and 'Skin'=3 would be specified as follows:
```
    V7
    N3
    C0.00 0.00 1.00 0.50 Bone
    C0.00 1.00 0.00 0.50 Fat
    C1.00 0.00 0.00 0.50 Skin
```

## Specifying a dataset via the 'dataset' option
Instead of providing the 'image_dir'/'labels_dir' pair, the training data can also be described by one or multiple json files. Example config that globs data from multiple json files:
```
{
    "dataset" = ["/dataA/dataset.json", "/dataB/dataset.json"],
    "output_dir" = "<path where trained model and logs are saved>",
    ...
}
```
The `dataset.json` loosely follows the convention used for the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) datasets, and popular codes e.g. [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md).
