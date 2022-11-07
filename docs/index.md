# Segmantic

Segmantic is a [MONAI]/[PyTorch]-based library for medical image segmentation.

## Features

- utilities to prepare datasets for use in training of segmentation and style transfer networks
- segmentation networks
- evaluation metrics (fast confusion matrix, [Hausdorff] distance, ...)
- image to image translation (based on [pix2pix / cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) to help generalization from limited training data

## Installation

Create a virtual environment

```sh
cd /your/working/directory
python -m venv .venv
```

Activate it using e.g. `source .venv/bin/activate` on Linux/Mac and `.venv\Scripts\activate.bat` on Windows.

To install this repo (this will install all dependencies):

```sh
pip install git+https://github.com/dyollb/segmantic.git#egg=segmantic
```

Or in edit/dev mode

```sh
pip install -e git+https://github.com/dyollb/segmantic.git#egg=segmantic[dev]
```

## Getting started

The project layout is as follows:

```bash
|-- segmantic/src/segmantic
    |-- commands   # command lines
    |-- i2i        # image-to-image translation (style transfer)
    |-- prepro     # module containing utils to prepare your data
    |-- seg        # semantic segmentation: training, inference and evaluation
    |-- util       # utility functions
|-- segmantic/pyproject.toml
|-- segmantic/scripts
|-- segmantic/tests
```

<!-- INVISIBLE REFERENCES BELOW THIS LINE. ORDER ALPHABETICALLY -->

[hausdorff]: https://en.wikipedia.org/wiki/Hausdorff_space
[monai]: https://monai.io/
[pytorch]: https://pytorch.org/
