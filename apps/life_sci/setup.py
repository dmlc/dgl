#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dgllife
import sys

from setuptools import find_packages

if '--inplace' in sys.argv:
    from distutils.core import setup
else:
    from setuptools import setup

setup(
    name='dgllife',
    version=dgllife.__version__,
    description='DGL-based package for Life Science',
    keywords=[
        'pytorch',
        'dgl',
        'graph-neural-networks',
        'life-science',
        'drug-discovery'
    ],
    zip_safe=False,
    maintainer='DGL Team',
    packages=[package for package in find_packages()
              if package.startswith('dgllife')],
    install_requires=[
        'dgl>=0.4',
        'torch>=1'
        'scikit-learn>=0.21.2',
        'pandas>=0.25.1',
        'requests>=2.22.0'
    ],
    url='https://github.com/dmlc/dgl/tree/master/apps/life_sci',
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
