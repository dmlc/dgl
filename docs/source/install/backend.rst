.. _backends:

Working with different backends
===============================

DGL supports PyTorch, MXNet and Tensorflow backends. 
DGL will choose the backend on the following options (high priority to low priority)
- `DGLBACKEND` environment
   - You can use `DGLBACKEND=[BACKEND] python gcn.py ...` to specify the backend
   - Or `export DGLBACKEND=[BACKEND]` to set the global environment variable 
- `config.json` file under "~/.dgl"
   - You can use `python -m dgl.backend.set_default_backend [BACKEND]` to set the default backend

Currently BACKEND can be chosen from mxnet, pytorch, tensorflow.

PyTorch backend
---------------

Export ``DGLBACKEND`` as ``pytorch`` to specify PyTorch backend. The required PyTorch
version is 1.1.0 or later. See `pytorch.org <https://pytorch.org>`_ for installation instructions.

MXNet backend
-------------

Export ``DGLBACKEND`` as ``mxnet`` to specify MXNet backend. The required MXNet version is
1.5 or later. See `mxnet.apache.org <https://mxnet.apache.org/get_started>`_ for installation
instructions.

MXNet uses uint32 as the default data type for integer tensors, which only supports graph of
size smaller than 2^32. To enable large graph training, *build* MXNet with ``USE_INT64_TENSOR_SIZE=1``
flag. See `this FAQ <https://mxnet.apache.org/api/faq/large_tensor_support>`_ for more information.

MXNet 1.5 and later has an option to enable Numpy shape mode for ``NDArray`` objects, some DGL models
need this mode to be enabled to run correctly. However, this mode may not compatible with pretrained
model parameters with this mode disabled, e.g. pretrained models from GluonCV and GluonNLP.
By setting ``DGL_MXNET_SET_NP_SHAPE``, users can switch this mode on or off.

Tensorflow backend
------------------

Export ``DGLBACKEND`` as ``tensorflow`` to specify Tensorflow backend. The required Tensorflow
version is 2.2.0 or later. See `tensorflow.org <https://www.tensorflow.org/install>`_ for installation
instructions. In addition, DGL will set ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``true`` to prevent Tensorflow take over the whole GPU memory:

.. code:: bash

   pip install "tensorflow>=2.2.0rc1"  # when using tensorflow cpu version

