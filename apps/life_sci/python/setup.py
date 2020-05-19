#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import find_packages
from setuptools import setup

CURRENT_DIR = os.path.dirname(__file__)

def get_lib_path():
    """Get library path, name and version"""
     # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, './dgllife/libinfo.py')
    libinfo = {'__file__': libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)
    version = libinfo['__version__']

    return version

VERSION = get_lib_path()

setup(
    name='dgllife',
    version=VERSION,
    description='DGL-based package for Life Science',
    keywords=[
        'pytorch',
        'dgl',
        'graph-neural-networks',
        'life-science',
        'drug-discovery'
    ],
    maintainer='DGL Team',
    packages=[package for package in find_packages()
              if package.startswith('dgllife')],
    install_requires=[
        'scikit-learn>=0.22.2',
        'pandas',
        'requests>=2.22.0',
        'tqdm',
        'numpy>=1.14.0',
        'scipy>=1.1.0',
        'networkx>=2.1',
    ],
    url='https://github.com/dmlc/dgl/tree/master/apps/life_sci',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License'
    ],
    license='APACHE'
)
