import os
import numpy as np
import dgl
import dgl.ops as OPS
import backend as F
import unittest
import torch
from functools import partial
from torch.utils.data import DataLoader
from collections import defaultdict
from collections.abc import Iterator, Mapping
from itertools import product
from test_utils import parametrize_idtype
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
@pytest.mark.parametrize('mode', ['node', 'edge', 'walk'])
def test_saint(num_workers, mode):
    g = dgl.data.CoraFullDataset()[0]

    if mode == 'node':
        budget = 100
    elif mode == 'edge':
        budget = 200
    elif mode == 'walk':
        budget = (3, 2)

    sampler = dgl.dataloading.SAINTSampler(mode, budget)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(100), sampler, num_workers=num_workers)
    assert len(dataloader) == 100
    for sg in dataloader:
        pass

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

def _check_dtype(data, dtype, attr_name):
    if isinstance(data, dict):
        for k, v in data.items():
            assert getattr(v, attr_name) == dtype
    elif isinstance(data, list):
        for v in data:
            assert getattr(v, attr_name) == dtype
    else:
        assert getattr(data, attr_name) == dtype

def _check_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            assert v.device == F.ctx()
    elif isinstance(data, list):
        for v in data:
            assert v.device == F.ctx()
    else:
        assert data.device == F.ctx()

@parametrize_idtype
@pytest.mark.parametrize('sampler_name', ['full', 'neighbor', 'neighbor2'])
@pytest.mark.parametrize('pin_graph', [None, 'cuda_indices', 'cpu_indices'])
def test_node_dataloader(idtype, sampler_name, pin_graph):
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4])).astype(idtype)
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())
    g1.ndata['label'] = F.copy_to(F.randn((g1.num_nodes(),)), F.cpu())
    indices = F.arange(0, g1.num_nodes(), idtype)
    if F.ctx() != F.cpu():
        if pin_graph:
            g1.create_formats_()
            g1.pin_memory_()
            if pin_graph == 'cpu_indices':
                indices = F.arange(0, g1.num_nodes(), idtype, F.cpu())
            elif pin_graph == 'cuda_indices':
                if F._default_context_str == 'gpu':
                    indices = F.arange(0, g1.num_nodes(), idtype, F.cuda())
                else:
                    return  # skip
        else:
            g1 = g1.to('cuda')

    use_uva = pin_graph is not None and F.ctx() != F.cpu()

    for num_workers in [0, 1, 2]:
        sampler = {
            'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
            'neighbor': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
            'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3])}[sampler_name]
        dataloader = dgl.dataloading.NodeDataLoader(
            g1, indices, sampler, device=F.ctx(),
            batch_size=g1.num_nodes(),
            num_workers=(num_workers if (pin_graph and F.ctx() == F.cpu()) else 0),
            use_uva=use_uva)
        for input_nodes, output_nodes, blocks in dataloader:
            _check_device(input_nodes)
            _check_device(output_nodes)
            _check_device(blocks)
            _check_dtype(input_nodes, idtype, 'dtype')
            _check_dtype(output_nodes, idtype, 'dtype')
            _check_dtype(blocks, idtype, 'idtype')
    if g1.is_pinned():
        g1.unpin_memory_()

    g2 = dgl.heterograph({
         ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0]),
         ('user', 'followed-by', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2]),
         ('user', 'play', 'game'): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
         ('game', 'played-by', 'user'): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5])
    }).astype(idtype)
    for ntype in g2.ntypes:
        g2.nodes[ntype].data['feat'] = F.copy_to(F.randn((g2.num_nodes(ntype), 8)), F.cpu())
    indices = {nty: F.arange(0, g2.num_nodes(nty)) for nty in g2.ntypes}
    if F.ctx() != F.cpu():
        if pin_graph:
            g2.create_formats_()
            g2.pin_memory_()
            if pin_graph == 'cpu_indices':
                indices = {nty: F.arange(0, g2.num_nodes(nty), idtype, F.cpu()) for nty in g2.ntypes}
            elif pin_graph == 'cuda_indices':
                if F._default_context_str == 'gpu':
                    indices = {nty: F.arange(0, g2.num_nodes(), idtype, F.cuda()) for nty in g2.ntypes}
                else:
                    return  # skip
        else:
            g2 = g2.to('cuda')

    batch_size = max(g2.num_nodes(nty) for nty in g2.ntypes)
    sampler = {
        'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
        'neighbor': dgl.dataloading.MultiLayerNeighborSampler([{etype: 3 for etype in g2.etypes}] * 2),
        'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3])}[sampler_name]

    dataloader = dgl.dataloading.NodeDataLoader(
        g2, {nty: g2.nodes(nty) for nty in g2.ntypes},
        sampler, device=F.ctx(), batch_size=batch_size,
        num_workers=(num_workers if (pin_graph and F.ctx() == F.cpu()) else 0),
        use_uva=use_uva)
    assert isinstance(iter(dataloader), Iterator)
    for input_nodes, output_nodes, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(output_nodes)
        _check_device(blocks)
        _check_dtype(input_nodes, idtype, 'dtype')
        _check_dtype(output_nodes, idtype, 'dtype')
        _check_dtype(blocks, idtype, 'idtype')

    if g2.is_pinned():
        g2.unpin_memory_()

@pytest.mark.parametrize('sampler_name', ['full', 'neighbor'])
@pytest.mark.parametrize('neg_sampler', [
    dgl.dataloading.negative_sampler.Uniform(2),
    dgl.dataloading.negative_sampler.GlobalUniform(15, False, 3),
    dgl.dataloading.negative_sampler.GlobalUniform(15, True, 3)])
@pytest.mark.parametrize('pin_graph', [False, True])
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

def _create_homogeneous():
    s = torch.randint(0, 200, (1000,), device=F.ctx())
    d = torch.randint(0, 200, (1000,), device=F.ctx())
    src = torch.cat([s, d])
    dst = torch.cat([d, s])
    g = dgl.graph((s, d), num_nodes=200)
    reverse_eids = torch.cat([torch.arange(1000, 2000), torch.arange(0, 1000)]).to(F.ctx())
    always_exclude = torch.randint(0, 1000, (50,), device=F.ctx())
    seed_edges = torch.arange(0, 1000, device=F.ctx())
    return g, reverse_eids, always_exclude, seed_edges

def _create_heterogeneous():
    edges = {}
    for utype, etype, vtype in [('A', 'AA', 'A'), ('A', 'AB', 'B')]:
        s = torch.randint(0, 200, (1000,), device=F.ctx())
        d = torch.randint(0, 200, (1000,), device=F.ctx())
        edges[utype, etype, vtype] = (s, d)
        edges[vtype, 'rev-' + etype, utype] = (d, s)
    g = dgl.heterograph(edges, num_nodes_dict={'A': 200, 'B': 200})
    reverse_etypes = {'AA': 'rev-AA', 'AB': 'rev-AB', 'rev-AA': 'AA', 'rev-AB': 'AB'}
    always_exclude = {
        'AA': torch.randint(0, 1000, (50,), device=F.ctx()),
        'AB': torch.randint(0, 1000, (50,), device=F.ctx())}
    seed_edges = {
        'AA': torch.arange(0, 1000, device=F.ctx()),
        'AB': torch.arange(0, 1000, device=F.ctx())}
    return g, reverse_etypes, always_exclude, seed_edges

def _find_edges_to_exclude(g, exclude, always_exclude, pair_eids):
    if exclude == None:
        return always_exclude
    elif exclude == 'self':
        return torch.cat([pair_eids, always_exclude]) if always_exclude is not None else pair_eids
    elif exclude == 'reverse_id':
        pair_eids = torch.cat([pair_eids, pair_eids + 1000])
        return torch.cat([pair_eids, always_exclude]) if always_exclude is not None else pair_eids
    elif exclude == 'reverse_types':
        pair_eids = {g.to_canonical_etype(k): v for k, v in pair_eids.items()}
        if ('A', 'AA', 'A') in pair_eids:
            pair_eids[('A', 'rev-AA', 'A')] = pair_eids[('A', 'AA', 'A')]
        if ('A', 'AB', 'B') in pair_eids:
            pair_eids[('B', 'rev-AB', 'A')] = pair_eids[('A', 'AB', 'B')]
        if always_exclude is not None:
            always_exclude = {g.to_canonical_etype(k): v for k, v in always_exclude.items()}
            for k in always_exclude.keys():
                if k in pair_eids:
                    pair_eids[k] = torch.cat([pair_eids[k], always_exclude[k]])
                else:
                    pair_eids[k] = always_exclude[k]
        return pair_eids

@pytest.mark.parametrize('always_exclude_flag', [False, True])
@pytest.mark.parametrize('exclude', [None, 'self', 'reverse_id', 'reverse_types'])
def test_edge_dataloader_excludes(exclude, always_exclude_flag):
    if exclude == 'reverse_types':
        g, reverse_etypes, always_exclude, seed_edges = _create_heterogeneous()
    else:
        g, reverse_eids, always_exclude, seed_edges = _create_homogeneous()
    g = g.to(F.ctx())
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    if not always_exclude_flag:
        always_exclude = None

    kwargs = {}
    kwargs['exclude'] = (
        partial(_find_edges_to_exclude, g, exclude, always_exclude) if always_exclude_flag
        else exclude)
    kwargs['reverse_eids'] = reverse_eids if exclude == 'reverse_id' else None
    kwargs['reverse_etypes'] = reverse_etypes if exclude == 'reverse_types' else None

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, seed_edges, sampler, batch_size=50, device=F.ctx(), **kwargs)
    for input_nodes, pair_graph, blocks in dataloader:
        block = blocks[0]
        pair_eids = pair_graph.edata[dgl.EID]
        block_eids = block.edata[dgl.EID]

        edges_to_exclude = _find_edges_to_exclude(g, exclude, always_exclude, pair_eids)
        if edges_to_exclude is None:
            continue
        edges_to_exclude = dgl.utils.recursive_apply(edges_to_exclude, lambda x: x.cpu().numpy())
        block_eids = dgl.utils.recursive_apply(block_eids, lambda x: x.cpu().numpy())

        if isinstance(edges_to_exclude, Mapping):
            for k in edges_to_exclude.keys():
                assert not np.isin(edges_to_exclude[k], block_eids[k]).any()
        else:
            assert not np.isin(edges_to_exclude, block_eids).any()

if __name__ == '__main__':
    test_node_dataloader(F.int32, 'neighbor', None)
