#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import shutil
import glob

from setuptools import find_packages
from setuptools.dist import Distribution
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    version = f.readline().strip()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_lib_path():
    """Get library path, name and version"""
     # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, './dgl/_ffi/libinfo.py')
    libinfo = {'__file__': libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)
    version = libinfo['__version__']
    if not os.getenv('CONDA_BUILD'):
        lib_path = libinfo['find_lib_path']()
        libs = [lib_path[0]]
    else:
        libs = None
    return libs, version

LIBS, VERSION = get_lib_path()

include_libs = False
wheel_include_libs = False
if not os.getenv('CONDA_BUILD'):
    if "bdist_wheel" in sys.argv:
        wheel_include_libs = True
    else:
        include_libs = True

setup_kwargs = {}

# For bdist_wheel only
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in LIBS:
            shutil.copy(path, os.path.join(CURRENT_DIR, 'dgl'))
            _, libname = os.path.split(path)
            print(libname)
            fo.write("include dgl/%s\n" % libname)
    setup_kwargs = {
        "include_package_data": True
    }

# For source tree setup
if include_libs:
    setup_kwargs = {
        "include_package_data": True,
        "data_files": [('dgl', LIBS)]
    }

setup(
    name='dgl',
    version=version,
    description='Deep Graph Library',
    zip_safe=False,
    maintainer='DGL Team',
    maintainer_email='wmjlyjemaine@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.1.0',
        'networkx>=2.1',
    ],
    url='https://github.com/jermainewang/dgl',
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    license='APACHE',
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIBS:
        _, libname = os.path.split(path)
        os.remove("dgl/%s" % libname)
