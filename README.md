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

## Example scripts

TODO
