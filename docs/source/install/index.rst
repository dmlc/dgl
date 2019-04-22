Install DGL
============

At this stage, we recommend installing DGL from ``conda`` or ``pip``.

System requirements
-------------------
Currently DGL is tested on

* Ubuntu 16.04
* macOS X
* Windows 10

DGL is expected to work on all Linux distributions later than Ubuntu 16.04, macOS X, and
Windows 10.

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
|         |         | command to install:                              |
|         |         |                                                  |
|         |         | .. code:: bash                                   |
|         |         |                                                  |
|         |         |    pip install --pre mxnet                       |
+---------+---------+--------------------------------------------------+
| numpy   | NumPy   | Does not support gradient computation            |
+---------+---------+--------------------------------------------------+

.. _install-from-source:

Install from source
-------------------
First, download the source files from GitHub:

.. code:: bash

   git clone --recursive https://github.com/dmlc/dgl.git

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

macOS
`````

Installation on macOS is similar to Linux. But macOS users need to install
building tools like clang, GNU Make, cmake first.

Tools like clang and GNU Make are packaged in **Command Line Tools** for macOS. To
install:

.. code:: bash

   xcode-select --install

To install other needed packages like cmake, we recommend first installing
**Homebrew**, which is a popular package manager for macOS. Detailed
instructions can be found on its `homepage <https://brew.sh/>`_.

After installation of Homebrew, install cmake by:

.. code:: bash

   brew install cmake

Then go to root directory of DGL repository, build shared library and
install Python binding for DGL:

.. code:: bash

   mkdir build
   cd build
   cmake ..
   make -j4
   cd ../python
   python setup.py install

We tested installation on macOS X with clang 10.0.0, GNU Make 3.81, and cmake
3.13.1.

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
   cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0 -DDGL_EXPORTS" -DCMAKE_MAKE_PROGRAM=mingw32-make .. -G "MSYS Makefiles"
   mingw32-make
   cd ..\python
   python setup.py install

We also support building DGL with MSBuild.  With `MS Build Tools <https://go.microsoft.com/fwlink/?linkid=840931>`_
and `CMake on Windows <https://cmake.org/download/>`_ installed, run the following
in VS2017 x64 Native tools command prompt:

.. code::

   MD build
   CD build
   cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64"
   msbuild dgl.sln
   cd ..\python
   python setup.py install
