"""Distributed dataloaders.
"""
from ...distributed import DistDataLoader
from ..dataloader import NodeCollator, EdgeCollator

def _remove_kwargs_dist(kwargs):
    if 'num_workers' in kwargs:
        del kwargs['num_workers']
    if 'pin_memory' in kwargs:
        del kwargs['pin_memory']
        print('Distributed DataLoaders do not support pin_memory.')
    return kwargs

class DistNodeDataLoader(DistDataLoader):
    def __init__(self, g, nids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
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
    def __init__(self, g, eids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
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
        self.dataloader = DistDataLoader(self.collator.dataset,
                                         collate_fn=self.collator.collate,
                                         **dataloader_kwargs)

        self.device = device
