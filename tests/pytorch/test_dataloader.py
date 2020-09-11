import dgl
import backend as F
import numpy as np
import unittest
from torch.utils.data import DataLoader
from collections import defaultdict
from itertools import product

def _check_neighbor_sampling_dataloader(g, nids, dl, mode):
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

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_neighbor_sampler_dataloader():
    g = dgl.heterograph({('user', 'follow', 'user'): ([0, 0, 0, 1, 1], [1, 2, 3, 3, 4])}, 
                        {'user': 6}).long()
    g = dgl.to_bidirected(g)
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
    }).long()
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
        _check_neighbor_sampling_dataloader(_g, nid, dl, mode)


if __name__ == '__main__':
    test_neighbor_sampler_dataloader()
