Install DGL-LifeSci
===================

This topic explains how to install DGL-LifeSci. We recommend installing DGL-LifeSci by using ``conda`` or ``pip``.

System requirements
-------------------
DGL-LifeSci works with the following operating systems:

* Ubuntu 16.04
* macOS X
* Windows 10

DGL-LifeSci requires:

* Python 3.6 or later
* `DGL 0.4.3 or later <https://www.dgl.ai/pages/start.html>`_
* `PyTorch 1.2.0 or later <https://pytorch.org/>`_

If you have just installed DGL, the first time you use it, a message will pop up as follows:

.. code:: bash

    DGL does not detect a valid backend option. Which backend would you like to work with?
    Backend choice (pytorch, mxnet or tensorflow):

and you need to enter ``pytorch``.

Additionally, we require **RDKit 2018.09.3** for cheminformatics. We recommend installing it with

.. code:: bash

    conda install -c conda-forge rdkit==2018.09.3

Other verions of RDKit are not tested.

Install from conda
----------------------
If ``conda`` is not yet installed, get either `miniconda <https://conda.io/miniconda.html>`_ or
the full `anaconda <https://www.anaconda.com/download/>`_.

.. code:: bash

    conda install -c dglteam dgllife

Install from pip
----------------

.. code:: bash

    pip install dgllife

.. _install-from-source:

Install from source
-------------------

To use the latest experimental features,

.. code:: bash

    git clone https://github.com/dmlc/dgl.git
    cd apps/life_sci/python
    python setup.py install
