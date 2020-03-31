#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dgllife

from setuptools import find_packages
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
        'torch>=1.1'
        'scikit-learn>=0.22.2',
        'pandas>=0.24.2',
        'requests>=2.22.0',
        'tqdm'
    ],
    url='https://github.com/dmlc/dgl/tree/master/apps/life_sci',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License'
    ],
    license='APACHE'
)
