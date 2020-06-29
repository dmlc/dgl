import dgl
import backend as F
import numpy as np
import unittest
from torch.utils.data import DataLoader
from collections import defaultdict

def _check_neighbor_sampling_dataloader(g, nids, dl):
    seeds = defaultdict(list)

    for input_nodes, output_nodes, blocks in dl:
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
                if prev_dst[utype] is not None:
                    assert F.array_equal(src, prev_dst[utype])
                u = src[uu]
                v = dst[vv]
                assert F.asnumpy(g.has_edges_between(u, v, etype=canonical_etype)).all()
                eid = block.edges[canonical_etype].data[dgl.EID]
                ufound, vfound = g.find_edges(eid, etype=canonical_etype)
                assert F.array_equal(ufound, u)
                assert F.array_equal(vfound, v)
            for ntype in block.dsttypes:
                src = block.srcnodes[ntype].data[dgl.NID]
                dst = block.dstnodes[ntype].data[dgl.NID]
                assert F.array_equal(src[:block.number_of_dst_nodes(ntype)], dst)
                prev_dst[ntype] = dst
        for ntype in blocks[-1].dsttypes:
            seeds[ntype].append(blocks[-1].dstnodes[ntype].data[dgl.NID])

    # Check if all nodes are iterated
    seeds = {k: F.cat(v, 0) for k, v in seeds.items()}
    for k, v in seeds.items():
        v_set = set(F.asnumpy(v))
        seed_set = set(nids[k])
        assert v_set == seed_set

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_neighbor_sampler_dataloader():
    g = dgl.graph([(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0)],
            'user', 'follow', num_nodes=6)
    g_sampler1 = dgl.sampling.MultiLayerNeighborSampler([2, 2], return_eids=True)
    g_sampler2 = dgl.sampling.MultiLayerNeighborSampler([None, None], return_eids=True)

    hg = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0)],
        ('user', 'plays', 'game'): [(0, 0), (1, 1), (1, 2), (3, 0), (5, 2)],
        ('game', 'wanted-by', 'user'): [(0, 1), (2, 1), (1, 3), (2, 3), (2, 5)]})
    hg_sampler1 = dgl.sampling.MultiLayerNeighborSampler(
        [{'plays': 1, 'wanted-by': 1, 'follow': 2}] * 2,
        return_eids=True)
    hg_sampler2 = dgl.sampling.MultiLayerNeighborSampler([None, None], return_eids=True)

    collators = [
        dgl.sampling.NodeCollator(g, [0, 1, 2, 3, 5], g_sampler1),
        dgl.sampling.NodeCollator(g, [4, 5], g_sampler1),
        dgl.sampling.NodeCollator(g, [0, 1, 2, 3, 5], g_sampler2),
        dgl.sampling.NodeCollator(g, [4, 5], g_sampler2),
        dgl.sampling.NodeCollator(hg, {'user': [0, 1, 3, 5], 'game': [0, 1, 2]}, hg_sampler1),
        dgl.sampling.NodeCollator(hg, {'user': [4, 5], 'game': [0, 1, 2]}, hg_sampler1),
        dgl.sampling.NodeCollator(hg, {'user': [0, 1, 3, 5], 'game': [0, 1, 2]}, hg_sampler2),
        dgl.sampling.NodeCollator(hg, {'user': [4, 5], 'game': [0, 1, 2]}, hg_sampler2)]
    nids = [
        {'user': [0, 1, 2, 3, 5]},
        {'user': [4, 5]},
        {'user': [0, 1, 2, 3, 5]},
        {'user': [4, 5]},
        {'user': [0, 1, 3, 5], 'game': [0, 1, 2]},
        {'user': [4, 5], 'game': [0, 1, 2]},
        {'user': [0, 1, 3, 5], 'game': [0, 1, 2]},
        {'user': [4, 5], 'game': [0, 1, 2]}]
    graphs = [g] * 4 + [hg] * 4
    samplers = [g_sampler1, g_sampler1, g_sampler2, g_sampler2, hg_sampler1, hg_sampler1, hg_sampler2, hg_sampler2]

    for _g, nid, collator in zip(graphs, nids, collators):
        dl = DataLoader(
            collator.dataset, collate_fn=collator.collate, batch_size=2, shuffle=True, drop_last=False)
        _check_neighbor_sampling_dataloader(_g, nid, dl)
    for _g, nid, sampler in zip(graphs, nids, samplers):
        dl = dgl.sampling.NodeDataLoader(_g, nid, sampler, batch_size=2, shuffle=True, drop_last=False)
        _check_neighbor_sampling_dataloader(_g, nid, dl)


if __name__ == '__main__':
    test_neighbor_sampler_dataloader()
