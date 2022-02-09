import os
import dgl
import dgl.ops as OPS
import backend as F
import unittest
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from collections.abc import Iterator
from itertools import product
import pytest


def test_graph_dataloader():
    batch_size = 16
    num_batches = 2
    minigc_dataset = dgl.data.MiniGCDataset(batch_size * num_batches, 10, 20)
    data_loader = dgl.dataloading.GraphDataLoader(minigc_dataset, batch_size=batch_size, shuffle=True)
    assert isinstance(iter(data_loader), Iterator)
    for graph, label in data_loader:
        assert isinstance(graph, dgl.DGLGraph)
        assert F.asnumpy(label).shape[0] == batch_size

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@pytest.mark.parametrize('num_workers', [0, 4])
def test_cluster_gcn(num_workers):
    dataset = dgl.data.CoraFullDataset()
    g = dataset[0]
    sampler = dgl.dataloading.ClusterGCNSampler(g, 100)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(100), sampler, batch_size=4, num_workers=num_workers)
    assert len(dataloader) == 25
    for i, sg in enumerate(dataloader):
        pass

@pytest.mark.parametrize('num_workers', [0, 4])
def test_shadow(num_workers):
    g = dgl.data.CoraFullDataset()[0]
    sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=5, shuffle=True, drop_last=False, num_workers=num_workers)
    for i, (input_nodes, output_nodes, subgraph) in enumerate(dataloader):
        assert torch.equal(input_nodes, subgraph.ndata[dgl.NID])
        assert torch.equal(input_nodes[:output_nodes.shape[0]], output_nodes)
        assert torch.equal(subgraph.ndata['label'], g.ndata['label'][input_nodes])
        assert torch.equal(subgraph.ndata['feat'], g.ndata['feat'][input_nodes])
        if i == 5:
            break


@pytest.mark.parametrize('num_workers', [0, 4])
def test_neighbor_nonuniform(num_workers):
    g = dgl.graph(([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]))
    g.edata['p'] = torch.FloatTensor([1, 1, 0, 0, 1, 1, 0, 0])
    sampler = dgl.dataloading.MultiLayerNeighborSampler([2], prob='p')
    dataloader = dgl.dataloading.NodeDataLoader(g, [0, 1], sampler, batch_size=1, device=F.ctx())
    for input_nodes, output_nodes, blocks in dataloader:
        seed = output_nodes.item()
        neighbors = set(input_nodes[1:].cpu().numpy())
        if seed == 1:
            assert neighbors == {5, 6}
        elif seed == 0:
            assert neighbors == {1, 2}

    g = dgl.heterograph({
        ('B', 'BA', 'A'): ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]),
        ('C', 'CA', 'A'): ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]),
        })
    g.edges['BA'].data['p'] = torch.FloatTensor([1, 1, 0, 0, 1, 1, 0, 0])
    g.edges['CA'].data['p'] = torch.FloatTensor([0, 0, 1, 1, 0, 0, 1, 1])
    sampler = dgl.dataloading.MultiLayerNeighborSampler([2], prob='p')
    dataloader = dgl.dataloading.NodeDataLoader(
        g, {'A': [0, 1]}, sampler, batch_size=1, device=F.ctx())
    for input_nodes, output_nodes, blocks in dataloader:
        seed = output_nodes['A'].item()
        # Seed and neighbors are of different node types so slicing is not necessary here.
        neighbors = set(input_nodes['B'].cpu().numpy())
        if seed == 1:
            assert neighbors == {5, 6}
        elif seed == 0:
            assert neighbors == {1, 2}

        neighbors = set(input_nodes['C'].cpu().numpy())
        if seed == 1:
            assert neighbors == {7, 8}
        elif seed == 0:
            assert neighbors == {3, 4}


def _check_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            assert v.device == F.ctx()
    elif isinstance(data, list):
        for v in data:
            assert v.device == F.ctx()
    else:
        assert data.device == F.ctx()

@pytest.mark.parametrize('sampler_name', ['full', 'neighbor', 'neighbor2'])
@pytest.mark.parametrize('pin_graph', [True, False])
def test_node_dataloader(sampler_name, pin_graph):
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]))
    if F.ctx() != F.cpu() and pin_graph:
        g1.create_formats_()
        g1.pin_memory_()
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())
    g1.ndata['label'] = F.copy_to(F.randn((g1.num_nodes(),)), F.cpu())

    for num_workers in [0, 1, 2]:
        sampler = {
            'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
            'neighbor': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
            'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3])}[sampler_name]
        dataloader = dgl.dataloading.NodeDataLoader(
            g1, g1.nodes(), sampler, device=F.ctx(),
            batch_size=g1.num_nodes(),
            num_workers=num_workers)
        for input_nodes, output_nodes, blocks in dataloader:
            _check_device(input_nodes)
            _check_device(output_nodes)
            _check_device(blocks)

    g2 = dgl.heterograph({
         ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0]),
         ('user', 'followed-by', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2]),
         ('user', 'play', 'game'): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
         ('game', 'played-by', 'user'): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5])
    })
    for ntype in g2.ntypes:
        g2.nodes[ntype].data['feat'] = F.copy_to(F.randn((g2.num_nodes(ntype), 8)), F.cpu())
    batch_size = max(g2.num_nodes(nty) for nty in g2.ntypes)
    sampler = {
        'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
        'neighbor': dgl.dataloading.MultiLayerNeighborSampler([{etype: 3 for etype in g2.etypes}] * 2),
        'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3])}[sampler_name]

    dataloader = dgl.dataloading.NodeDataLoader(
        g2, {nty: g2.nodes(nty) for nty in g2.ntypes},
        sampler, device=F.ctx(), batch_size=batch_size)
    assert isinstance(iter(dataloader), Iterator)
    for input_nodes, output_nodes, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(output_nodes)
        _check_device(blocks)

    if g1.is_pinned():
        g1.unpin_memory_()

@pytest.mark.parametrize('sampler_name', ['full', 'neighbor'])
@pytest.mark.parametrize('neg_sampler', [
    dgl.dataloading.negative_sampler.Uniform(2),
    dgl.dataloading.negative_sampler.GlobalUniform(15, False, 3),
    dgl.dataloading.negative_sampler.GlobalUniform(15, True, 3)])
@pytest.mark.parametrize('pin_graph', [True, False])
def test_edge_dataloader(sampler_name, neg_sampler, pin_graph):
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]))
    if F.ctx() != F.cpu() and pin_graph:
        g1.create_formats_()
        g1.pin_memory_()
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())

    sampler = {
        'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
        'neighbor': dgl.dataloading.MultiLayerNeighborSampler([3, 3])}[sampler_name]

    # no negative sampler
    dataloader = dgl.dataloading.EdgeDataLoader(
        g1, g1.edges(form='eid'), sampler, device=F.ctx(), batch_size=g1.num_edges())
    for input_nodes, pos_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(blocks)

    # negative sampler
    dataloader = dgl.dataloading.EdgeDataLoader(
        g1, g1.edges(form='eid'), sampler, device=F.ctx(),
        negative_sampler=neg_sampler, batch_size=g1.num_edges())
    for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(neg_pair_graph)
        _check_device(blocks)

    g2 = dgl.heterograph({
         ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0]),
         ('user', 'followed-by', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2]),
         ('user', 'play', 'game'): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
         ('game', 'played-by', 'user'): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5])
    })
    for ntype in g2.ntypes:
        g2.nodes[ntype].data['feat'] = F.copy_to(F.randn((g2.num_nodes(ntype), 8)), F.cpu())
    batch_size = max(g2.num_edges(ety) for ety in g2.canonical_etypes)
    sampler = {
        'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
        'neighbor': dgl.dataloading.MultiLayerNeighborSampler([{etype: 3 for etype in g2.etypes}] * 2),
        }[sampler_name]

    # no negative sampler
    dataloader = dgl.dataloading.EdgeDataLoader(
        g2, {ety: g2.edges(form='eid', etype=ety) for ety in g2.canonical_etypes},
        sampler, device=F.ctx(), batch_size=batch_size)
    for input_nodes, pos_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(blocks)

    # negative sampler
    dataloader = dgl.dataloading.EdgeDataLoader(
        g2, {ety: g2.edges(form='eid', etype=ety) for ety in g2.canonical_etypes},
        sampler, device=F.ctx(), negative_sampler=neg_sampler,
        batch_size=batch_size)

    assert isinstance(iter(dataloader), Iterator)
    for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(neg_pair_graph)
        _check_device(blocks)

    if g1.is_pinned():
        g1.unpin_memory_()

if __name__ == '__main__':
    test_graph_dataloader()
    test_cluster_gcn(0)
    test_neighbor_nonuniform(0)
    for sampler in ['full', 'neighbor']:
        test_node_dataloader(sampler)
        for neg_sampler in [
                dgl.dataloading.negative_sampler.Uniform(2),
                dgl.dataloading.negative_sampler.GlobalUniform(2, False),
                dgl.dataloading.negative_sampler.GlobalUniform(2, True)]:
            for pin_graph in [True, False]:
                test_edge_dataloader(sampler, neg_sampler, pin_graph)
