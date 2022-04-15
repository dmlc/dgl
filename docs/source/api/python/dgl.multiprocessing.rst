.. _apimultiprocessing:

dgl.multiprocessing
===================

This is a minimal wrapper of Python's native :mod:`multiprocessing` module.
It modifies the :class:`multiprocessing.Process` class to make forking
work with OpenMP in the DGL core library.

The API usage is exactly the same as the native module, so DGL does not provide
additional documentation.

In addition, if your backend is PyTorch, this module will also be compatible with
:mod:`torch.multiprocessing` module.

.. currentmodule:: dgl.multiprocessing.pytorch
.. autosummary::
    :toctree: ../../generated/

    call_once_and_share
    shared_tensor
