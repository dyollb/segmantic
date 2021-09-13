# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

import re
import sys
from pathlib import Path
from subprocess import check_output

from setuptools import setup, find_packages


def read_reqs(reqs_path: Path):
    return re.findall(
        r"(^[^#\n-][\w\[,\]]+[-~>=<.\w]*)", reqs_path.read_text(), re.MULTILINE
    )


def get_cuda_version():
    out = check_output(["nvidia-smi"]).decode("utf-8").split("\n")

    for line in out:
        if "CUDA Version:" in line:
            line = line.rsplit("CUDA Version:", 1)[1].strip()
            version = line.split(" ", 1)[0].strip().split(".")
            return [int(v) for v in version]
    return [0, 0]


current_dir = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve().parent
cuda_version = get_cuda_version()

if cuda_version[0] == 11:
    install_requirements = read_reqs(current_dir / "requirements" / "cuda_111.txt")
else:
    install_requirements = read_reqs(current_dir / "requirements" / "generic.txt")

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name='segmantic',
    version='0.1.0',
    description='Collection of tools for ML-based semantic image segmentation',
    long_description=readme,
    author='Bryn Lloyd',
    author_email='lloyd@itis.swiss',
    url='https://github.com/dyollb/segmantic.git',
    license=license,
    packages=find_packages(where="src"),
    package_dir={
        "": "src",
    },
)
