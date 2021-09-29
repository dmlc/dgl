# cython: language_level=3
from distutils.core import setup
from Cython.Build import cythonize
import numpy

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules=cythonize(["edge_sampler.pyx"]), include_dirs=[numpy.get_include()])
