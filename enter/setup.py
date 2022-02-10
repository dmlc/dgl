#!/usr/bin/env python

from setuptools import find_packages
from distutils.core import setup

setup(name='dglenter',
      version='0.0.1',
      description='DGL',
      author='DGL Team',
      author_email='wmjlyjemaine@gmail.com',
      packages=find_packages(),
      install_requires=[
          'typer>=0.4.0',
          'isort>=5.10.1',
          'autopep8>=1.6.0',
          'numpydoc>=1.1.0',
          "pydantic>=1.9.0",
          "ruamel.yaml>=0.17.20"
      ],
    license='APACHE',
      entry_points={
          'console_scripts': [
              "dgl-enter = dglenter.cli.cli:main"
          ]
      },
      url='https://github.com/dmlc/dgl',
      )
