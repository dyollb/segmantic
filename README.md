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
python scripts\run_monai_unet.py train-config -c config.txt --print-defaults
```

Edit `config.txt` e.g. to
```
{
    "image_dir": "work/inputs/images",
    "labels_dir": "work/inputs/labels",
    "tissue_list": "work/inputs/labels_16.txt",
    "output_dir": "work/outputs",
    "checkpoint_file": null,
    "num_channels": 1,
    "spatial_dims": 3,
    "spatial_size": null,
    "max_epochs": 5,
    "augment_intensity": false,
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
python scripts\run_monai_unet.py train-config -c config.txt
```
