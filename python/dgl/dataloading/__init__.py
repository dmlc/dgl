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
from .. import backend as F
from .neighbor_sampler import *
from .cluster_gcn import *
from .shadow import *
from .base import *
from . import negative_sampler
if F.get_preferred_backend() == 'pytorch':
    from .dataloader import *
    from .dist_dataloader import *
