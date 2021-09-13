# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
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

