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

def _check_neighbor_sampling_dataloader(g, nids, dl, mode, collator):
    seeds = defaultdict(list)

    for item in dl:
        if mode == 'node':
            input_nodes, output_nodes, blocks = item
        elif mode == 'edge':
            input_nodes, pair_graph, blocks = item
            output_nodes = pair_graph.ndata[dgl.NID]
        elif mode == 'link':
            input_nodes, pair_graph, neg_graph, blocks = item
            output_nodes = pair_graph.ndata[dgl.NID]
            for ntype in pair_graph.ntypes:
                assert F.array_equal(pair_graph.nodes[ntype].data[dgl.NID], neg_graph.nodes[ntype].data[dgl.NID])

        if len(g.ntypes) > 1:
            for ntype in g.ntypes:
                assert F.array_equal(input_nodes[ntype], blocks[0].srcnodes[ntype].data[dgl.NID])
                assert F.array_equal(output_nodes[ntype], blocks[-1].dstnodes[ntype].data[dgl.NID])
        else:
            assert F.array_equal(input_nodes, blocks[0].srcdata[dgl.NID])
            assert F.array_equal(output_nodes, blocks[-1].dstdata[dgl.NID])

        prev_dst = {ntype: None for ntype in g.ntypes}
        for block in blocks:
            for canonical_etype in block.canonical_etypes:
                utype, etype, vtype = canonical_etype
                uu, vv = block.all_edges(order='eid', etype=canonical_etype)
                src = block.srcnodes[utype].data[dgl.NID]
                dst = block.dstnodes[vtype].data[dgl.NID]
                assert F.array_equal(
                    block.srcnodes[utype].data['feat'], g.nodes[utype].data['feat'][src])
                assert F.array_equal(
                    block.dstnodes[vtype].data['feat'], g.nodes[vtype].data['feat'][dst])
                if prev_dst[utype] is not None:
                    assert F.array_equal(src, prev_dst[utype])
                u = src[uu]
                v = dst[vv]
                assert F.asnumpy(g.has_edges_between(u, v, etype=canonical_etype)).all()
                eid = block.edges[canonical_etype].data[dgl.EID]
                assert F.array_equal(
                    block.edges[canonical_etype].data['feat'],
                    g.edges[canonical_etype].data['feat'][eid])
                ufound, vfound = g.find_edges(eid, etype=canonical_etype)
                assert F.array_equal(ufound, u)
                assert F.array_equal(vfound, v)
            for ntype in block.dsttypes:
                src = block.srcnodes[ntype].data[dgl.NID]
                dst = block.dstnodes[ntype].data[dgl.NID]
                assert F.array_equal(src[:block.number_of_dst_nodes(ntype)], dst)
                prev_dst[ntype] = dst

        if mode == 'node':
            for ntype in blocks[-1].dsttypes:
                seeds[ntype].append(blocks[-1].dstnodes[ntype].data[dgl.NID])
        elif mode == 'edge' or mode == 'link':
            for etype in pair_graph.canonical_etypes:
                seeds[etype].append(pair_graph.edges[etype].data[dgl.EID])

    # Check if all nodes/edges are iterated
    seeds = {k: F.cat(v, 0) for k, v in seeds.items()}
    for k, v in seeds.items():
        if k in nids:
            seed_set = set(F.asnumpy(nids[k]))
        elif isinstance(k, tuple) and k[1] in nids:
            seed_set = set(F.asnumpy(nids[k[1]]))
        else:
            continue

        v_set = set(F.asnumpy(v))
        assert v_set == seed_set

def test_neighbor_sampler_dataloader():
    g = dgl.heterograph({('user', 'follow', 'user'): ([0, 0, 0, 1, 1], [1, 2, 3, 3, 4])},
                        {'user': 6}).long()
    g = dgl.to_bidirected(g).to(F.ctx())
    g.ndata['feat'] = F.randn((6, 8))
    g.edata['feat'] = F.randn((10, 4))
    reverse_eids = F.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=F.int64)
    g_sampler1 = dgl.dataloading.MultiLayerNeighborSampler([2, 2], return_eids=True)
    g_sampler2 = dgl.dataloading.MultiLayerFullNeighborSampler(2, return_eids=True)

    hg = dgl.heterograph({
         ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0]),
         ('user', 'followed-by', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2]),
         ('user', 'play', 'game'): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
         ('game', 'played-by', 'user'): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5])
    }).long().to(F.ctx())
    for ntype in hg.ntypes:
        hg.nodes[ntype].data['feat'] = F.randn((hg.number_of_nodes(ntype), 8))
    for etype in hg.canonical_etypes:
        hg.edges[etype].data['feat'] = F.randn((hg.number_of_edges(etype), 4))
    hg_sampler1 = dgl.dataloading.MultiLayerNeighborSampler(
        [{'play': 1, 'played-by': 1, 'follow': 2, 'followed-by': 1}] * 2, return_eids=True)
    hg_sampler2 = dgl.dataloading.MultiLayerFullNeighborSampler(2, return_eids=True)
    reverse_etypes = {'follow': 'followed-by', 'followed-by': 'follow', 'play': 'played-by', 'played-by': 'play'}

    collators = []
    graphs = []
    nids = []
    modes = []
    for seeds, sampler in product(
            [F.tensor([0, 1, 2, 3, 5], dtype=F.int64), F.tensor([4, 5], dtype=F.int64)],
            [g_sampler1, g_sampler2]):
        collators.append(dgl.dataloading.NodeCollator(g, seeds, sampler))
        graphs.append(g)
        nids.append({'user': seeds})
        modes.append('node')

        collators.append(dgl.dataloading.EdgeCollator(g, seeds, sampler))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('edge')

        collators.append(dgl.dataloading.EdgeCollator(
            g, seeds, sampler, exclude='self'))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('edge')

        collators.append(dgl.dataloading.EdgeCollator(
            g, seeds, sampler, exclude='reverse_id', reverse_eids=reverse_eids))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('edge')

        collators.append(dgl.dataloading.EdgeCollator(
            g, seeds, sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(2)))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('link')

        collators.append(dgl.dataloading.EdgeCollator(
            g, seeds, sampler, exclude='self', negative_sampler=dgl.dataloading.negative_sampler.Uniform(2)))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('link')

        collators.append(dgl.dataloading.EdgeCollator(
            g, seeds, sampler, exclude='reverse_id', reverse_eids=reverse_eids,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(2)))
        graphs.append(g)
        nids.append({'follow': seeds})
        modes.append('link')

    for seeds, sampler in product(
            [{'user': F.tensor([0, 1, 3, 5], dtype=F.int64), 'game': F.tensor([0, 1, 2], dtype=F.int64)},
             {'user': F.tensor([4, 5], dtype=F.int64), 'game': F.tensor([0, 1, 2], dtype=F.int64)}],
            [hg_sampler1, hg_sampler2]):
        collators.append(dgl.dataloading.NodeCollator(hg, seeds, sampler))
        graphs.append(hg)
        nids.append(seeds)
        modes.append('node')

    for seeds, sampler in product(
            [{'follow': F.tensor([0, 1, 3, 5], dtype=F.int64), 'play': F.tensor([1, 3], dtype=F.int64)},
             {'follow': F.tensor([4, 5], dtype=F.int64), 'play': F.tensor([1, 3], dtype=F.int64)}],
            [hg_sampler1, hg_sampler2]):
        collators.append(dgl.dataloading.EdgeCollator(hg, seeds, sampler))
        graphs.append(hg)
        nids.append(seeds)
        modes.append('edge')

        collators.append(dgl.dataloading.EdgeCollator(
            hg, seeds, sampler, exclude='reverse_types', reverse_etypes=reverse_etypes))
        graphs.append(hg)
        nids.append(seeds)
        modes.append('edge')

        collators.append(dgl.dataloading.EdgeCollator(
            hg, seeds, sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(2)))
        graphs.append(hg)
        nids.append(seeds)
        modes.append('link')

        collators.append(dgl.dataloading.EdgeCollator(
            hg, seeds, sampler, exclude='reverse_types', reverse_etypes=reverse_etypes,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(2)))
        graphs.append(hg)
        nids.append(seeds)
        modes.append('link')

    for _g, nid, collator, mode in zip(graphs, nids, collators, modes):
        dl = DataLoader(
            collator.dataset, collate_fn=collator.collate, batch_size=2, shuffle=True, drop_last=False)
        assert isinstance(iter(dl), Iterator)
        _check_neighbor_sampling_dataloader(_g, nid, dl, mode, collator)

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
    sgiter = dgl.dataloading.ClusterGCNSubgraphIterator(g, 100, '.', refresh=True)
    dataloader = dgl.dataloading.GraphDataLoader(sgiter, batch_size=4, num_workers=num_workers)
    for sg in dataloader:
        assert sg.batch_size == 4

    sgiter = dgl.dataloading.ClusterGCNSubgraphIterator(g, 100, '.', refresh=False) # use cache
    dataloader = dgl.dataloading.GraphDataLoader(sgiter, batch_size=4, num_workers=num_workers)
    for sg in dataloader:
        assert sg.batch_size == 4

@pytest.mark.parametrize('num_workers', [0, 4])
def test_shadow(num_workers):
    g = dgl.data.CoraFullDataset()[0]
    sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=5, shuffle=True, drop_last=False, num_workers=num_workers)
    for i, (input_nodes, output_nodes, (subgraph,)) in enumerate(dataloader):
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

@pytest.mark.parametrize('sampler_name', ['full', 'neighbor', 'neighbor2', 'shadow'])
def test_node_dataloader(sampler_name):
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]))
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())
    g1.ndata['label'] = F.copy_to(F.randn((g1.num_nodes(),)), F.cpu())

    for load_input, load_output in [(None, None), ({'feat': g1.ndata['feat']}, {'label': g1.ndata['label']})]:
        for async_load in [False, True]:
            for num_workers in [0, 1, 2]:
                sampler = {
                    'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
                    'neighbor': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
                    'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
                    'shadow': dgl.dataloading.ShaDowKHopSampler([3, 3])}[sampler_name]
                dataloader = dgl.dataloading.NodeDataLoader(
                    g1, g1.nodes(), sampler, device=F.ctx(),
                    load_input=load_input,
                    load_output=load_output,
                    async_load=async_load,
                    batch_size=g1.num_nodes(),
                    num_workers=num_workers)
                for input_nodes, output_nodes, blocks in dataloader:
                    _check_device(input_nodes)
                    _check_device(output_nodes)
                    _check_device(blocks)
                    if load_input:
                        _check_device(blocks[0].srcdata['feat'])
                        OPS.copy_u_sum(blocks[0], blocks[0].srcdata['feat'])
                    if load_output:
                        _check_device(blocks[-1].dstdata['label'])
                        OPS.copy_u_sum(blocks[-1], blocks[-1].dstdata['label'])

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
        'neighbor2': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
        'shadow': dgl.dataloading.ShaDowKHopSampler([{etype: 3 for etype in g2.etypes}] * 2)}[sampler_name]

    for async_load in [False, True]:
        dataloader = dgl.dataloading.NodeDataLoader(
            g2, {nty: g2.nodes(nty) for nty in g2.ntypes},
            sampler, device=F.ctx(), async_load=async_load, batch_size=batch_size)
        assert isinstance(iter(dataloader), Iterator)
        for input_nodes, output_nodes, blocks in dataloader:
            _check_device(input_nodes)
            _check_device(output_nodes)
            _check_device(blocks)

    status = False
    try:
        dgl.dataloading.NodeDataLoader(
            g2, {nty: g2.nodes(nty) for nty in g2.ntypes},
            sampler, device=F.ctx(), load_input={'feat': g1.ndata['feat']}, batch_size=batch_size)
    except dgl.DGLError:
        status = True
    assert status


@pytest.mark.parametrize('sampler_name', ['full', 'neighbor', 'shadow'])
def test_edge_dataloader(sampler_name):
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)

    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]))
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())

    sampler = {
        'full': dgl.dataloading.MultiLayerFullNeighborSampler(2),
        'neighbor': dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
        'shadow': dgl.dataloading.ShaDowKHopSampler([3, 3])}[sampler_name]

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
        'shadow': dgl.dataloading.ShaDowKHopSampler([{etype: 3 for etype in g2.etypes}] * 2)}[sampler_name]

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

if __name__ == '__main__':
    test_neighbor_sampler_dataloader()
    test_graph_dataloader()
    test_cluster_gcn(0)
    test_neighbor_nonuniform(0)
    for sampler in ['full', 'neighbor', 'shadow']:
        test_node_dataloader(sampler)
        test_edge_dataloader(sampler)
