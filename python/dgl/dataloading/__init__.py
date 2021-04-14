"""The ``dgl.dataloading`` package contains:

* Data loader classes for iterating over a set of nodes or edges in a graph and generates
  computation dependency via neighborhood sampling methods.

* Various sampler classes that perform neighborhood sampling for multi-layer GNNs.

* Negative samplers for link prediction.

For a holistic explanation on how different components work together.
Read the user guide :ref:`guide-minibatch`.

.. note::
    This package is experimental and the interfaces may be subject
    to changes in future releases. It currently only has implementations in PyTorch.
"""
from .neighbor import *
from .dataloader import *

from . import negative_sampler
from .async_transferer import AsyncTransferer

from .. import backend as F

if F.get_preferred_backend() == 'pytorch':
    from .pytorch import *
