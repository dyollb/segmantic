# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='segmania',
    version='0.1.0',
    description='Collection of tools for ML-based semantic image segmentation',
    long_description=readme,
    author='Bryn Lloyd',
    author_email='lloyd@itis.swiss',
    url='https://git.speag.com/lloyd/segmania.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

