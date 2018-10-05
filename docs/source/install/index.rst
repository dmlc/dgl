Install DGL
============

At this stage, we recommend installing DGL from source. To quickly try out DGL and its demo/tutorials, checkout `Install from docker`_.

Get source codes
----------------
First, download the source files from github. Note you need to use the ``--recursive`` option to
also clone the submodules.

.. code:: bash

    git clone --recursive https://github.com/jermainewang/dgl.git

You can also clone the repository first and type following commands:

.. code:: bash

    git submodule init
    git submodule update


Build shared library
--------------------
Before building the library, please make sure the following dependencies are installed
(use ubuntu as an example):

.. code:: bash

    sudo apt-get update
    sudo apt-get install -y python

We use cmake (minimal version 2.8) to build the library.

.. code:: bash

    mkdir build
    cd build
    cmake ..
    make -j4

Build python binding
--------------------
DGL's python binding depends on following packages (tested version):

* numpy (>= 1.14.0)
* scipy (>= 1.1.0)
* networkx (>= 2.1)

To install them, use following command:

.. code:: bash

    pip install --user numpy scipy networkx

There are several ways to setup DGL's python binding. We recommend developers at the current stage
use environment variables to find python packages.

.. code:: bash

    export DGL_HOME=/path/to/dgl
    export PYTHONPATH=$DGL_HOME$/python:${PYTHONPATH}
    export DGL_LIBRARY_PATH=$DGL_HOME$/build

The ``DGL_LIBRARY_PATH`` variable is used for our python package to locate the shared library
built above. Use following command to test whether the installation is successful or not.

.. code:: bash

    python -c 'import dgl'

Install from docker
-------------------
TBD
