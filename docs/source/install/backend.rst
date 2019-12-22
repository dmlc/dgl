Working with different backends
===============================

DGL supports PyTorch, MXNet and Tensorflow backends. To change them, set the ``DGLBACKEND``
environcment variable. The default backend is PyTorch.

PyTorch backend
---------------

Export ``DGLBACKEND`` as ``pytorch`` to specify PyTorch backend. The required PyTorch
version is 0.4.1 or later. See `pytorch.org <https://pytorch.org>`_ for installation instructions.

MXNet backend
-------------

Export ``DGLBACKEND`` as ``mxnet`` to specify MXNet backend. The required MXNet version is
1.5 or later. See `mxnet.apache.org <https://mxnet.apache.org/get_started>`_ for installation
instructions.

MXNet uses uint32 as the default data type for integer tensors, which only supports graph of
size smaller than 2^32. To enable large graph training, *build* MXNet with ``USE_INT64_TENSOR_SIZE=1``
flag. See `this FAQ <https://mxnet.apache.org/api/faq/large_tensor_support>`_ for more information.

Tensorflow backend
------------------

Export ``DGLBACKEND`` as ``tensorflow`` to specify Tensorflow backend. The required Tensorflow
version is 2.0 or later. See `tensorflow.org <https://www.tensorflow.org/install>`_ for installation
instructions. In addition, Tensorflow backend requires ``tfdlpack`` package installed as follows:

.. code:: bash

   pip install tfdlpack  # when using tensorflow

or

.. code:: bash

   pip install tfdlpack-gpu  # when using tensorflow-gpu
