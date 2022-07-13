"""Distributed dataloaders.
"""
import inspect
from ..distributed import DistDataLoader
# Still depends on the legacy NodeCollator...
from .._dataloading.dataloader import NodeCollator, EdgeCollator

def _remove_kwargs_dist(kwargs):
    if 'num_workers' in kwargs:
        del kwargs['num_workers']
    if 'pin_memory' in kwargs:
        del kwargs['pin_memory']
        print('Distributed DataLoaders do not support pin_memory.')
    return kwargs

class DistNodeDataLoader(DistDataLoader):
    """Sampled graph data loader over nodes for distributed graph storage.

    It wraps an iterable over a set of nodes, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch, on
    a distributed graph.

    All the arguments have the same meaning as the single-machine counterpart
    :class:`dgl.dataloading.DataLoader` except the first argument
    :attr:`g` which must be a :class:`dgl.distributed.DistGraph`.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.

    nids, graph_sampler, device, kwargs :
        See :class:`dgl.dataloading.DataLoader`.

    See also
    --------
    dgl.dataloading.DataLoader
    """
    def __init__(self, g, nids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(NodeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        if device is None:
            # for the distributed case default to the CPU
            device = 'cpu'
        assert device == 'cpu', 'Only cpu is supported in the case of a DistGraph.'
        # Distributed DataLoader currently does not support heterogeneous graphs
        # and does not copy features.  Fallback to normal solution
        self.collator = NodeCollator(g, nids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(self.collator.dataset,
                         collate_fn=self.collator.collate,
                         **dataloader_kwargs)
        self.device = device

class DistEdgeDataLoader(DistDataLoader):
    """Sampled graph data loader over edges for distributed graph storage.

    It wraps an iterable over a set of edges, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch for
    edge classification, edge regression, and link prediction, on a distributed
    graph.

    All the arguments have the same meaning as the single-machine counterpart
    :class:`dgl.dataloading.EdgeDataLoader` except the first argument
    :attr:`g` which must be a :class:`dgl.distributed.DistGraph`.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.

    eids, graph_sampler, device, kwargs :
        See :class:`dgl.dataloading.EdgeDataLoader`.

    See also
    --------
    dgl.dataloading.EdgeDataLoader
    """
    def __init__(self, g, eids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(EdgeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        if device is None:
            # for the distributed case default to the CPU
            device = 'cpu'
        assert device == 'cpu', 'Only cpu is supported in the case of a DistGraph.'
        # Distributed DataLoader currently does not support heterogeneous graphs
        # and does not copy features.  Fallback to normal solution
        self.collator = EdgeCollator(g, eids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(self.collator.dataset,
                         collate_fn=self.collator.collate,
                         **dataloader_kwargs)

        self.device = device
