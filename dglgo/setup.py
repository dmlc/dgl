#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dglgo",
    version="0.0.2",
    description="DGL",
    author="DGL Team",
    author_email="wmjlyjemaine@gmail.com",
    packages=find_packages(),
    install_requires=[
        "typer>=0.4.0",
        "isort>=5.10.1",
        "autopep8>=1.6.0",
        "numpydoc>=1.1.0",
        "pydantic>=1.9.0",
        "ruamel.yaml>=0.17.20",
        "PyYAML>=5.1",
        "ogb>=1.3.3",
        "rdkit-pypi",
        "scikit-learn>=0.20.0",
    ],
    package_data={"": ["./*"]},
    include_package_data=True,
    license="APACHE",
    entry_points={"console_scripts": ["dgl = dglgo.cli.cli:main"]},
    url="https://github.com/dmlc/dgl",
)
