Install DGL
============

This topic explains how to install DGL. We recommend installing DGL by using ``conda`` or ``pip``.

System requirements
-------------------
DGL works with the following operating systems:

* Ubuntu 16.04
* macOS X
* Windows 10

DGL requires Python version 3.5 or later. Python 3.4 or earlier is not
tested. Python 2 support is coming.

DGL supports multiple tensor libraries as backends, e.g., PyTorch, MXNet. For requirements on backends and how to select one, see
`Working with different backends`_.

Starting at version 0.3, DGL is separated into CPU and CUDA builds.  The builds share the
same Python package name. If you install DGL with a CUDA 9 build after you install the
CPU build, then the CPU build is overwritten.

Install from conda
----------------------
If ``conda`` is not yet installed, get either `miniconda <https://conda.io/miniconda.html>`_ or
the full `anaconda <https://www.anaconda.com/download/>`_.

With ``conda`` installed, you will want install DGL into Python 3.5 ``conda`` environment.
Run `conda create -n dgl python=3.5` to create the environment.
Activate the environment by running `source activate dgl`.
After the ``conda`` environment is activated, run one of the following commands.

.. code:: bash

   conda install -c dglteam dgl              # For CPU Build
   conda install -c dglteam dgl-cuda9.0      # For CUDA 9.0 Build
   conda install -c dglteam dgl-cuda10.0     # For CUDA 10.0 Build

Install from pip
----------------
For CPU builds, run the following command to install with ``pip``.

.. code:: bash

   pip install dgl
   
For CUDA builds, run one of the following commands and specify the CUDA version.

.. code:: bash

   pip install dgl           # For CPU Build
   pip install dgl-cu90      # For CUDA 9.0 Build
   pip install dgl-cu92      # For CUDA 9.2 Build
   pip install dgl-cu100     # For CUDA 10.0 Build

For the most current nightly build from master branch, run one of the following commands.

.. code:: bash

   pip install --pre dgl           # For CPU Build
   pip install --pre dgl-cu90      # For CUDA 9.0 Build
   pip install --pre dgl-cu92      # For CUDA 9.2 Build
   pip install --pre dgl-cu100     # For CUDA 10.0 Build


Working with different backends
-------------------------------

DGL supports PyTorch and MXNet. Here's how to change them.

Switching backend
`````````````````

The backend is controlled by ``DGLBACKEND`` environment variable, which defaults to
``pytorch``.  The following values are supported.

+---------+---------+--------------------------------------------------+
| Value   | Backend | Constraints                                      |
+=========+=========+==================================================+
| pytorch | PyTorch | Requires 0.4.1 or later. See                     |
|         |         | `pytorch.org <https://pytorch.org>`_             |
+---------+---------+--------------------------------------------------+
| mxnet   | MXNet   | Requires either MXNet 1.5 for CPU                   |
|         |         |                                                  |
|         |         | .. code:: bash                                   |
|         |         |                                                  |
|         |         |    pip install mxnet                             |
|         |         |                                                  |
|         |         | or MXNet for GPU with CUDA version, e.g. for CUDA 9.2               |
|         |         |                                                  |
|         |         | .. code:: bash                                   |
|         |         |                                                  |
|         |         |    pip install mxnet-cu90                        |
|         |         |                                                  |
+---------+---------+--------------------------------------------------+
| numpy   | NumPy   | Does not support gradient computation            |
+---------+---------+--------------------------------------------------+

.. _install-from-source:

Install from source
-------------------
Download the source files from GitHub.

.. code:: bash

   git clone --recursive https://github.com/dmlc/dgl.git

(Optional) Clone the repository first, and then run the following:

.. code:: bash

   git submodule init
   git submodule update

Linux
`````

Install the system packages for building the shared library. For Debian and Ubuntu
users, run:

.. code:: bash

   sudo apt-get update
   sudo apt-get install -y build-essential python3-dev make cmake

For Fedora/RHEL/CentOS users, run:

.. code:: bash

   sudo yum install -y gcc-c++ python3-devel make cmake

Build the shared library. Use the configuration template ``cmake/config.cmake``.
Copy it to either the project directory or the build directory and change the
configuration as you wish. For example, change ``USE_CUDA`` to ``ON`` will
enable a CUDA build. You could also pass ``-DKEY=VALUE`` to the cmake command
for the same purpose.

- CPU-only build
   .. code:: bash

      mkdir build
      cd build
      cmake ..
      make -j4
- CUDA build
   .. code:: bash

      mkdir build
      cd build
      cmake -DUSE_CUDA=ON ..
      make -j4

Finally, install the Python binding.

.. code:: bash

   cd ../python
   python setup.py install

macOS
`````

Installation on macOS is similar to Linux. But macOS users need to install build tools like clang, GNU Make, and cmake first. These installation steps were tested on macOS X with clang 10.0.0, GNU Make 3.81, and cmake 3.13.1.

Tools like clang and GNU Make are packaged in **Command Line Tools** for macOS. To
install, run the following:

.. code:: bash

   xcode-select --install

To install other needed packages like cmake, we recommend first installing
**Homebrew**, which is a popular package manager for macOS. To learn more, see the `Homebrew website <https://brew.sh/>`_.

After you install Homebrew, install cmake.

.. code:: bash

   brew install cmake

Go to root directory of the DGL repository, build a shared library, and
install the Python binding for DGL.

.. code:: bash

   mkdir build
   cd build
   cmake -DUSE_OPENMP=off ..
   make -j4
   cd ../python
   python setup.py install

Windows
```````

The Windows source build is tested with CMake and MinGW/GCC.  We highly recommend
using CMake and GCC from `conda installations <https://conda.io/miniconda.html>`_.  To
get started, run the following:

.. code:: bash

   conda install cmake m2w64-gcc m2w64-make

Build the shared library and install the Python binding.

.. code::

   md build
   cd build
   cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0 -DDGL_EXPORTS" -DCMAKE_MAKE_PROGRAM=mingw32-make .. -G "MSYS Makefiles"
   mingw32-make
   cd ..\python
   python setup.py install

You can also build DGL with MSBuild.  With `MS Build Tools <https://go.microsoft.com/fwlink/?linkid=840931>`_
and `CMake on Windows <https://cmake.org/download/>`_ installed, run the following
in VS2017 x64 Native tools command prompt.

.. code::

   MD build
   CD build
   cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64"
   msbuild dgl.sln
   cd ..\python
   python setup.py install
