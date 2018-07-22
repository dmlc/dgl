#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os.path

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    version = f.readline().strip()

setuptools.setup(
    name='dgl',
    version=version,
    description='Deep Graph Library',
    maintainer='DGL Team',
    maintainer_email='wmjlyjemaine@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.1.0',
        'networkx>=2.1',
    ],
    data_files=[('', ['VERSION'])],
    url='https://github.com/jermainewang/dgl-1')
