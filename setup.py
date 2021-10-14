# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

import re
import sys
from pathlib import Path
from subprocess import check_output
from contextlib import suppress
from typing import Set

from setuptools import setup, find_packages


def read_reqs(reqs_path: Path):
    return re.findall(
        r"(^[^#\n-][\w\[,\]]+[-~>=<.\w]*)", reqs_path.read_text(), re.MULTILINE
    )


def get_cuda_version():
    with suppress(UnicodeDecodeError, FileNotFoundError):
        out = check_output(["nvidia-smi"]).decode("utf-8")
        match = re.search(r"CUDA Version:\s+(\d+).(\d+)", out)
        if match:
            version = tuple(int(c) for c in match.groups())
            print("\nDetected CUDA version: %s.%s\n" % (version[0], version[1]))
            return version

    print("\nCould not detected CUDA version\n", file=sys.stderr)
    return (0, 0)


def get_extra_requires(path: Path, add_all: bool = True):
    import re
    from collections import defaultdict

    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags: Set[str] = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps["all"] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


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
    name="segmantic",
    version="0.1.0",
    description="Collection of tools for ML-based semantic image segmentation",
    long_description=readme,
    author="Bryn Lloyd",
    author_email="lloyd@itis.swiss",
    url="https://github.com/dyollb/segmantic.git",
    license=license,
    install_requires=install_requirements,
    extras_require=get_extra_requires(current_dir / "extras_require.txt"),
    packages=find_packages(where="src"),
    package_dir={
        "": "src",
    },
)
