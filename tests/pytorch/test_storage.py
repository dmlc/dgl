import os
import numpy as np
import dgl
import pytest
import torch

class CustomGraphStorageWrapper(object):
    def __init__(self, g, ndata, edata):
        self.g = g
        self.ndata = ndata
        self.edata = edata

    def get_node_storage(self, key, ntype=None):
        return self.ndata[ntype][key]

    def get_edge_storage(self, key, etype=None):
        return self.edata[self.g.to_canonical_etype(etype)][key]

    def __getattr__(self, attr, default=None):
        return getattr(self.g, attr, default)

@pytest.mark.parametrize('pin_prefetcher', [None, True, False])
@pytest.mark.parametrize('use_prefetch_thread', [None, True, False])
@pytest.mark.parametrize('use_alternate_streams', [None, True, False])
def test_custom_graph_storage(pin_prefetcher, use_prefetch_thread, use_alternate_streams):
    if any(v is None for v in [pin_prefetcher, use_prefetch_thread, use_alternate_streams):
        # Do we want to test where not all of them are None?
        pin_prefetcher = use_prefetch_thread = use_alternate_streams = None
    g = dgl.data.CoraFullDataset()[0]
    ndata = {
            g.ntypes[0]:
            {key: dgl.storages.TensorStorage(value) for key, value in g.ndata.items()}}
    edata = {
            g.canonical_etypes[0]:
            {key: dgl.storages.TensorStorage(value) for key, value in g.edata.items()}}
    g_wrapped = CustomGraphStorageWrapper(g, ndata, edata)
    sampler = dgl.dataloading.NeighborSampler(
            [2, 2], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    dataloader = dgl.dataloading.DataLoader(
            g_wrapped, torch.arange(g.num_nodes()), sampler, batch_size=40)
    for input_nodes, output_nodes, blocks in dataloader:
        pass

if __name__ == '__main__':
    test_custom_graph_storage()
