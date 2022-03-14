import re
import sys
from subprocess import check_output
from contextlib import suppress

from setuptools import setup


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


if __name__ == "__main__":
    setup()
