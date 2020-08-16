"""Classes that involves iterating over nodes or edges in a graph and generates
computation dependency of necessary nodes with neighborhood sampling methods.

This includes

* :py:class:`~dgl.dataloading.pytorch.NodeDataLoader` for iterating over the nodes in
  a graph in minibatches.

* :py:class:`~dgl.dataloading.pytorch.EdgeDataLoader` for iterating over the edges in
  a graph in minibatches.

* Various sampler classes that perform neighborhood sampling for multi-layer GNNs.

* Negative samplers for link prediction.

NOTE: this module is experimental and the interfaces may be subject to changes in
future releases.
"""
from .neighbor import *
from .dataloader import *

from . import negative_sampler

from .. import backend as F

if F.get_preferred_backend() == 'pytorch':
    from .pytorch import *
