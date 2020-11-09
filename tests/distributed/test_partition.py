import dgl
import sys
import os
import numpy as np
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from dgl.heterograph_index import create_unitgraph_from_coo
from dgl.distributed import partition_graph, load_partition
from dgl import function as fn
import backend as F
import unittest
import pickle
import random

def _get_inner_node_mask(graph, ntype_id):
    if dgl.NTYPE in graph.ndata:
        return graph.ndata['inner_node'] * (graph.ndata[dgl.NTYPE] == ntype_id) == 1
    else:
        return graph.ndata['inner_node'] == 1

def _get_inner_edge_mask(graph, etype_id):
    if dgl.ETYPE in graph.edata:
        return graph.edata['inner_edge'] * (graph.edata[dgl.ETYPE] == etype_id) == 1
    else:
        return graph.edata['inner_edge'] == 1

def _get_part_ranges(id_ranges):
    if isinstance(id_ranges, dict):
        return {key:np.concatenate([np.array(l) for l in id_ranges[key]]).reshape(-1, 2) \
                for key in id_ranges}
    else:
        return np.concatenate([np.array(l) for l in id_range[key]]).reshape(-1, 2)


def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo', random_state=100) != 0).astype(np.int64)
    return dgl.from_scipy(arr)

def create_random_hetero():
    num_nodes = {'n1': 10000, 'n2': 10010, 'n3': 10020}
    etypes = [('n1', 'r1', 'n2'),
              ('n1', 'r2', 'n3'),
              ('n2', 'r3', 'n3')]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype], num_nodes[dst_ntype], density=0.001, format='coo',
                          random_state=100)
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, num_nodes)

def verify_hetero_graph(g, parts):
    num_nodes = {ntype:0 for ntype in g.ntypes}
    num_edges = {etype:0 for etype in g.etypes}
    for part in parts:
        assert len(g.ntypes) == len(F.unique(part.ndata[dgl.NTYPE]))
        assert len(g.etypes) == len(F.unique(part.edata[dgl.ETYPE]))
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            inner_node_mask = _get_inner_node_mask(part, ntype_id)
            num_inner_nodes = F.sum(F.astype(inner_node_mask, F.int64), 0)
            num_nodes[ntype] += num_inner_nodes
        for etype in g.etypes:
            etype_id = g.get_etype_id(etype)
            inner_edge_mask = _get_inner_edge_mask(part, etype_id)
            num_inner_edges = F.sum(F.astype(inner_edge_mask, F.int64), 0)
            num_edges[etype] += num_inner_edges
    # Verify the number of nodes are correct.
    for ntype in g.ntypes:
        print('node {}: {}, {}'.format(ntype, g.number_of_nodes(ntype), num_nodes[ntype]))
        assert g.number_of_nodes(ntype) == num_nodes[ntype]
    # Verify the number of edges are correct.
    for etype in g.etypes:
        print('edge {}: {}, {}'.format(etype, g.number_of_edges(etype), num_edges[etype]))
        assert g.number_of_edges(etype) == num_edges[etype]

    nids = {ntype:[] for ntype in g.ntypes}
    eids = {etype:[] for etype in g.etypes}
    for part in parts:
        src, dst, eid = part.edges(form='all')
        orig_src = part.ndata['orig_id'][src]
        orig_dst = part.ndata['orig_id'][dst]
        orig_eid = part.edata['orig_id'][eid]
        etype_arr = part.edata[dgl.ETYPE][eid]
        eid_type = part.edata[dgl.EID][eid]
        for etype in g.etypes:
            etype_id = g.get_etype_id(etype)
            src1 = orig_src[etype_arr == etype_id]
            dst1 = orig_dst[etype_arr == etype_id]
            eid1 = orig_eid[etype_arr == etype_id]
            exist = g.has_edges_between(src1, dst1, etype=etype)
            assert np.all(exist.detach().numpy())
            eid2 = g.edge_ids(src1, dst1, etype=etype)
            assert np.all((eid1 == eid2).detach().numpy())
            eids[etype].append(eid_type[etype_arr == etype_id])
            # Make sure edge Ids fall into a range.
            inner_edge_mask = _get_inner_edge_mask(part, etype_id)
            inner_eids = np.sort(part.edata[dgl.EID][inner_edge_mask].numpy())
            assert np.all(inner_eids == np.arange(inner_eids[0], inner_eids[-1] + 1))

        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            # Make sure inner nodes have Ids fall into a range.
            inner_node_mask = _get_inner_node_mask(part, ntype_id)
            inner_nids = part.ndata[dgl.NID][inner_node_mask]
            assert np.all(F.asnumpy(inner_nids == F.arange(inner_nids[0], inner_nids[-1] + 1)))
            nids[ntype].append(inner_nids)

    for ntype in nids:
        nids_type = F.cat(nids[ntype], 0)
        uniq_ids = F.unique(nids_type)
        # We should get all nodes.
        assert len(uniq_ids) == g.number_of_nodes(ntype)
    for etype in eids:
        eids_type = F.cat(eids[etype], 0)
        uniq_ids = F.unique(eids_type)
        assert len(uniq_ids) == g.number_of_edges(etype)
    # TODO(zhengda) this doesn't check 'part_id'

def verify_graph_feats(g, part, node_feats):
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        for name in g.nodes[ntype].data:
            if name in [dgl.NID, 'inner_node']:
                continue
            inner_node_mask = part.ndata['inner_node'] * (part.ndata[dgl.NTYPE] == ntype_id) == 1
            inner_nids = part.ndata[dgl.NID][inner_node_mask]
            min_nids = F.min(inner_nids, 0)
            true_feats = g.nodes[ntype].data[name][part.ndata['orig_id'][inner_node_mask]]
            assert np.all((node_feats[ntype + '/' + name][inner_nids - min_nids] \
                           == true_feats).detach().numpy())

def check_hetero_partition(hg, part_method):
    hg.nodes['n1'].data['labels'] = F.arange(0, hg.number_of_nodes('n1'))
    hg.nodes['n1'].data['feats'] = F.tensor(np.random.randn(hg.number_of_nodes('n1'), 10), F.float32)
    hg.edges['r1'].data['feats'] = F.tensor(np.random.randn(hg.number_of_edges('r1'), 10), F.float32)
    num_parts = 4
    num_hops = 1

    partition_graph(hg, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                    part_method=part_method, reshuffle=True)
    parts = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, ntypes, etypes = load_partition('/tmp/partition/test.json', i)
        parts.append(part_g)
        verify_graph_feats(hg, part_g, node_feats)
    verify_hetero_graph(hg, parts)

def check_partition(g, part_method, reshuffle):
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10), F.float32)
    g.edata['feats'] = F.tensor(np.random.randn(g.number_of_edges(), 10), F.float32)
    g.update_all(fn.copy_src('feats', 'msg'), fn.sum('msg', 'h'))
    g.update_all(fn.copy_edge('feats', 'msg'), fn.sum('msg', 'eh'))
    num_parts = 4
    num_hops = 2

    partition_graph(g, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                    part_method=part_method, reshuffle=reshuffle)
    part_sizes = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, ntypes, etypes = load_partition('/tmp/partition/test.json', i)

        # Check the metadata
        assert gpb._num_nodes() == g.number_of_nodes()
        assert gpb._num_edges() == g.number_of_edges()

        assert gpb.num_partitions() == num_parts
        gpb_meta = gpb.metadata()
        assert len(gpb_meta) == num_parts
        assert len(gpb.partid2nids(i)) == gpb_meta[i]['num_nodes']
        assert len(gpb.partid2eids(i)) == gpb_meta[i]['num_edges']
        part_sizes.append((gpb_meta[i]['num_nodes'], gpb_meta[i]['num_edges']))

        local_nid = gpb.nid2localnid(F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node']), i)
        assert F.dtype(local_nid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
        local_eid = gpb.eid2localeid(F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge']), i)
        assert F.dtype(local_eid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_eid) == np.arange(0, len(local_eid)))

        # Check the node map.
        local_nodes = F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node'])
        llocal_nodes = F.nonzero_1d(part_g.ndata['inner_node'])
        local_nodes1 = gpb.partid2nids(i)
        assert F.dtype(local_nodes1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_nodes)) == np.sort(F.asnumpy(local_nodes1)))

        # Check the edge map.
        local_edges = F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge'])
        local_edges1 = gpb.partid2eids(i)
        assert F.dtype(local_edges1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_edges)) == np.sort(F.asnumpy(local_edges1)))

        if reshuffle:
            part_g.ndata['feats'] = F.gather_row(g.ndata['feats'], part_g.ndata['orig_id'])
            part_g.edata['feats'] = F.gather_row(g.edata['feats'], part_g.edata['orig_id'])
            # when we read node data from the original global graph, we should use orig_id.
            local_nodes = F.boolean_mask(part_g.ndata['orig_id'], part_g.ndata['inner_node'])
            local_edges = F.boolean_mask(part_g.edata['orig_id'], part_g.edata['inner_edge'])
        else:
            part_g.ndata['feats'] = F.gather_row(g.ndata['feats'], part_g.ndata[dgl.NID])
            part_g.edata['feats'] = F.gather_row(g.edata['feats'], part_g.edata[dgl.NID])
        part_g.update_all(fn.copy_src('feats', 'msg'), fn.sum('msg', 'h'))
        part_g.update_all(fn.copy_edge('feats', 'msg'), fn.sum('msg', 'eh'))
        assert F.allclose(F.gather_row(g.ndata['h'], local_nodes),
                          F.gather_row(part_g.ndata['h'], llocal_nodes))
        assert F.allclose(F.gather_row(g.ndata['eh'], local_nodes),
                          F.gather_row(part_g.ndata['eh'], llocal_nodes))

        for name in ['labels', 'feats']:
            assert '_N/' + name in node_feats
            assert node_feats['_N/' + name].shape[0] == len(local_nodes)
            assert np.all(F.asnumpy(g.ndata[name])[F.asnumpy(local_nodes)] == F.asnumpy(node_feats['_N/' + name]))
        for name in ['feats']:
            assert '_E/' + name in edge_feats
            assert edge_feats['_E/' + name].shape[0] == len(local_edges)
            assert np.all(F.asnumpy(g.edata[name])[F.asnumpy(local_edges)] == F.asnumpy(edge_feats['_E/' + name]))

    if reshuffle:
        node_map = []
        edge_map = []
        for i, (num_nodes, num_edges) in enumerate(part_sizes):
            node_map.append(np.ones(num_nodes) * i)
            edge_map.append(np.ones(num_edges) * i)
        node_map = np.concatenate(node_map)
        edge_map = np.concatenate(edge_map)
        nid2pid = gpb.nid2partid(F.arange(0, len(node_map)))
        assert F.dtype(nid2pid) in (F.int32, F.int64)
        assert np.all(F.asnumpy(nid2pid) == node_map)
        eid2pid = gpb.eid2partid(F.arange(0, len(edge_map)))
        assert F.dtype(eid2pid) in (F.int32, F.int64)
        assert np.all(F.asnumpy(eid2pid) == edge_map)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_partition():
    g = create_random_graph(10000)
    check_partition(g, 'metis', True)
    check_partition(g, 'random', True)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_hetero_partition():
    hg = create_random_hetero()
    check_hetero_partition(hg, 'metis')
    check_hetero_partition(hg, 'random')


if __name__ == '__main__':
    os.makedirs('/tmp/partition', exist_ok=True)
    test_hetero_partition()
    test_partition()
