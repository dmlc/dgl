#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dglchem
import sys

from setuptools import find_packages

if '--inplace' in sys.argv:
    from distutils.core import setup
else:
    from setuptools import setup

setup(
    name='dglchem',
    version=dglchem.__version__,
    description='DGL-based package for Chemistry',
    keywords=[
        'pytorch',
        'dgl',
        'graph-neural-networks',
        'chemistry',
        'drug-discovery'
    ],
    zip_safe=False,
    maintainer='DGL Team',
    packages=find_packages(),
    install_requires=[
        'dgl>=0.4',
        'torch>=1.2.0'
        'scikit-learn>=0.21.2',
        'pandas>=0.25.1',
        'requests>=2.22.0'
    ],
    url='https://github.com/dmlc/dgl/tree/master/apps/chem',
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
