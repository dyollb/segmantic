# Segmantic

Segmantic is a [PyTorch](https://pytorch.org/)-based library for medical image segmentation.

## Features

* utilities to prepare datasets for use in training of segmentation and style transfer networks
* image to image translation (based on [pix2pix / cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) to help generalization from limited training data
* segmentation networks

## Installation

Create a virtal environment
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

## Getting started

The project layout is as follows:

    setup.py
    src/segmantic
        prepro/     # module containing utils to prepare your data
        i2i/        # image-to-image translation (style transfer)
        seg/        # semantic segmentation networks, training and prediction
    scripts/
    tests/
