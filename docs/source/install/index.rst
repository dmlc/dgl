Install DGL
============

At this stage, we recommend installing DGL from ``conda`` or ``pip``.

System requirements
-------------------
Currently DGL is tested on

* Ubuntu 16.04
* OS X
* Windows 7

DGL is expected to work on all Linux distributions later than Ubuntu 16.04, OS X, and
Windows 7 or later.

DGL also requires the Python version to be 3.5 or later.  Python 3.4 or less is not
tested, and Python 2 support is coming.

DGL supports multiple tensor libraries (e.g. PyTorch, MXNet) as backends; refer
`Working with different backends`_ for requirements on backends and how to select a
backend.

Install from conda
----------------------
One can either grab `miniconda <https://conda.io/miniconda.html>`_ or
the full `anaconda <https://www.anaconda.com/download/>`_ if ``conda``
has not been installed.

Once the conda environment is activated, run

.. code:: bash

   conda install -c dglteam dgl

Install from pip
----------------
One can simply run the following command to install via ``pip``:

.. code:: bash

   pip install dgl

Working with different backends
-------------------------------

Currently DGL supports PyTorch and MXNet.

Switching backend
`````````````````

The backend is controlled by ``DGLBACKEND`` environment variable, which defaults to
``pytorch``.  Currently it supports the following values:

+---------+---------+--------------------------------------------------+
| Value   | Backend | Memo                                             |
+=========+=========+==================================================+
| pytorch | PyTorch | Requires 0.4.1 or later; see                     |
|         |         | `official website <https://pytorch.org>`_        |
+---------+---------+--------------------------------------------------+
| mxnet   | MXNet   | Requires nightly build; run the following        |
|         |         | command to install (TODO):                       |
|         |         |                                                  |
|         |         | .. code:: bash                                   |
|         |         |                                                  |
|         |         |    pip install --pre mxnet                       |
+---------+---------+--------------------------------------------------+
| numpy   | NumPy   | Does not support gradient computation            |
+---------+---------+--------------------------------------------------+

Install from source
-------------------
First, download the source files from GitHub:

.. code:: bash

   git clone --recursive https://github.com/jermainewang/dgl.git

One can also clone the repository first and run the following:

.. code:: bash

   git submodule init
   git submodule update

Linux
`````

Install the system packages for building the shared library, for Debian/Ubuntu
users, run:

.. code:: bash

   sudo apt-get update
   sudo apt-get install -y build-essential build-dep python3-dev make cmake

For Fedora/RHEL/CentOS users, run:

.. code:: bash

   sudo yum install -y gcc-c++ python3-devel make cmake

Build the shared library and install the Python binding afterwards:

.. code:: bash

   mkdir build
   cd build
   cmake ..
   make -j4
   cd ../python
   python setup.py install

OSX
```

TODO

Windows
```````

Currently Windows source build is tested with CMake and MinGW/GCC.  We highly recommend
using CMake and GCC from `conda installations <https://conda.io/miniconda.html>`_.  To
do so, run

.. code:: bash

   conda install cmake m2w64-gcc m2w64-make

Then build the shared library and install the Python binding:

.. code::

   md build
   cd build
   cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0 -DTVM_EXPORTS" -DCMAKE_MAKE_PROGRAM=mingw32-make .. -G "MSYS Makefiles"
   mingw32-make
   cd ..\python
   python setup.py install
