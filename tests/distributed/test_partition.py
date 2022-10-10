import dgl
import sys
import os
import numpy as np
from scipy import sparse as spsp
from dgl.distributed import partition_graph, load_partition, load_partition_feats
from dgl.distributed.graph_partition_book import BasicPartitionBook, RangePartitionBook, \
    NodePartitionPolicy, EdgePartitionPolicy, HeteroDataName
from dgl import function as fn
import backend as F
import unittest
import tempfile
from utils import reset_envs
from dgl.distributed.partition import RESERVED_FIELD_DTYPE

def _verify_partition_data_types(part_g):
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in part_g.ndata:
            assert part_g.ndata[k].dtype == dtype
        if k in part_g.edata:
            assert part_g.edata[k].dtype == dtype

def _get_inner_node_mask(graph, ntype_id):
    if dgl.NTYPE in graph.ndata:
        dtype = F.dtype(graph.ndata['inner_node'])
        return graph.ndata['inner_node'] * F.astype(graph.ndata[dgl.NTYPE] == ntype_id, dtype) == 1
    else:
        return graph.ndata['inner_node'] == 1

def _get_inner_edge_mask(graph, etype_id):
    if dgl.ETYPE in graph.edata:
        dtype = F.dtype(graph.edata['inner_edge'])
        return graph.edata['inner_edge'] * F.astype(graph.edata[dgl.ETYPE] == etype_id, dtype) == 1
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
    num_nodes = {'n1': 1000, 'n2': 1010, 'n3': 1020}
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
        _, _, eid = part.edges(form='all')
        etype_arr = F.gather_row(part.edata[dgl.ETYPE], eid)
        eid_type = F.gather_row(part.edata[dgl.EID], eid)
        for etype in g.etypes:
            etype_id = g.get_etype_id(etype)
            eids[etype].append(F.boolean_mask(eid_type, etype_arr == etype_id))
            # Make sure edge Ids fall into a range.
            inner_edge_mask = _get_inner_edge_mask(part, etype_id)
            inner_eids = np.sort(F.asnumpy(F.boolean_mask(part.edata[dgl.EID], inner_edge_mask)))
            assert np.all(inner_eids == np.arange(inner_eids[0], inner_eids[-1] + 1))

        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            # Make sure inner nodes have Ids fall into a range.
            inner_node_mask = _get_inner_node_mask(part, ntype_id)
            inner_nids = F.boolean_mask(part.ndata[dgl.NID], inner_node_mask)
            assert np.all(F.asnumpy(inner_nids == F.arange(F.as_scalar(inner_nids[0]),
                                                           F.as_scalar(inner_nids[-1]) + 1)))
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

def verify_graph_feats(g, gpb, part, node_feats, edge_feats, orig_nids, orig_eids):
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        inner_node_mask = _get_inner_node_mask(part, ntype_id)
        inner_nids = F.boolean_mask(part.ndata[dgl.NID],inner_node_mask)
        ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
        partid = gpb.nid2partid(inner_type_nids, ntype)
        assert np.all(F.asnumpy(ntype_ids) == ntype_id)
        assert np.all(F.asnumpy(partid) == gpb.partid)

        orig_id = orig_nids[ntype][inner_type_nids]
        local_nids = gpb.nid2localnid(inner_type_nids, gpb.partid, ntype)

        for name in g.nodes[ntype].data:
            if name in [dgl.NID, 'inner_node']:
                continue
            true_feats = F.gather_row(g.nodes[ntype].data[name], orig_id)
            ndata = F.gather_row(node_feats[ntype + '/' + name], local_nids)
            assert np.all(F.asnumpy(ndata == true_feats))

    for etype in g.etypes:
        etype_id = g.get_etype_id(etype)
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = F.boolean_mask(part.edata[dgl.EID],inner_edge_mask)
        etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
        partid = gpb.eid2partid(inner_type_eids, etype)
        assert np.all(F.asnumpy(etype_ids) == etype_id)
        assert np.all(F.asnumpy(partid) == gpb.partid)

        orig_id = orig_eids[etype][inner_type_eids]
        local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

        for name in g.edges[etype].data:
            if name in [dgl.EID, 'inner_edge']:
                continue
            true_feats = F.gather_row(g.edges[etype].data[name], orig_id)
            edata = F.gather_row(edge_feats[etype + '/' + name], local_eids)
            assert np.all(F.asnumpy(edata == true_feats))

def check_hetero_partition(hg, part_method, num_parts=4, num_trainers_per_machine=1, load_feats=True):
    hg.nodes['n1'].data['labels'] = F.arange(0, hg.number_of_nodes('n1'))
    hg.nodes['n1'].data['feats'] = F.tensor(np.random.randn(hg.number_of_nodes('n1'), 10), F.float32)
    hg.edges['r1'].data['feats'] = F.tensor(np.random.randn(hg.number_of_edges('r1'), 10), F.float32)
    hg.edges['r1'].data['labels'] = F.arange(0, hg.number_of_edges('r1'))
    num_hops = 1

    orig_nids, orig_eids = partition_graph(hg, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                                           part_method=part_method, reshuffle=True, return_mapping=True,
                                           num_trainers_per_machine=num_trainers_per_machine)
    assert len(orig_nids) == len(hg.ntypes)
    assert len(orig_eids) == len(hg.etypes)
    for ntype in hg.ntypes:
        assert len(orig_nids[ntype]) == hg.number_of_nodes(ntype)
    for etype in hg.etypes:
        assert len(orig_eids[etype]) == hg.number_of_edges(etype)
    parts = []
    shuffled_labels = []
    shuffled_elabels = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, ntypes, etypes = load_partition(
            '/tmp/partition/test.json', i, load_feats=load_feats)
        _verify_partition_data_types(part_g)
        if not load_feats:
            assert not node_feats
            assert not edge_feats
            node_feats, edge_feats = load_partition_feats('/tmp/partition/test.json', i)
        if num_trainers_per_machine > 1:
            for ntype in hg.ntypes:
                name = ntype + '/trainer_id'
                assert name in node_feats
                part_ids = F.floor_div(node_feats[name], num_trainers_per_machine)
                assert np.all(F.asnumpy(part_ids) == i)

            for etype in hg.etypes:
                name = etype + '/trainer_id'
                assert name in edge_feats
                part_ids = F.floor_div(edge_feats[name], num_trainers_per_machine)
                assert np.all(F.asnumpy(part_ids) == i)
        # Verify the mapping between the reshuffled IDs and the original IDs.
        # These are partition-local IDs.
        part_src_ids, part_dst_ids = part_g.edges()
        # These are reshuffled global homogeneous IDs.
        part_src_ids = F.gather_row(part_g.ndata[dgl.NID], part_src_ids)
        part_dst_ids = F.gather_row(part_g.ndata[dgl.NID], part_dst_ids)
        part_eids = part_g.edata[dgl.EID]
        # These are reshuffled per-type IDs.
        src_ntype_ids, part_src_ids = gpb.map_to_per_ntype(part_src_ids)
        dst_ntype_ids, part_dst_ids = gpb.map_to_per_ntype(part_dst_ids)
        etype_ids, part_eids = gpb.map_to_per_etype(part_eids)
        # These are original per-type IDs.
        for etype_id, etype in enumerate(hg.etypes):
            part_src_ids1 = F.boolean_mask(part_src_ids, etype_ids == etype_id)
            src_ntype_ids1 = F.boolean_mask(src_ntype_ids, etype_ids == etype_id)
            part_dst_ids1 = F.boolean_mask(part_dst_ids, etype_ids == etype_id)
            dst_ntype_ids1 = F.boolean_mask(dst_ntype_ids, etype_ids == etype_id)
            part_eids1 = F.boolean_mask(part_eids, etype_ids == etype_id)
            assert np.all(F.asnumpy(src_ntype_ids1 == src_ntype_ids1[0]))
            assert np.all(F.asnumpy(dst_ntype_ids1 == dst_ntype_ids1[0]))
            src_ntype = hg.ntypes[F.as_scalar(src_ntype_ids1[0])]
            dst_ntype = hg.ntypes[F.as_scalar(dst_ntype_ids1[0])]
            orig_src_ids1 = F.gather_row(orig_nids[src_ntype], part_src_ids1)
            orig_dst_ids1 = F.gather_row(orig_nids[dst_ntype], part_dst_ids1)
            orig_eids1 = F.gather_row(orig_eids[etype], part_eids1)
            orig_eids2 = hg.edge_ids(orig_src_ids1, orig_dst_ids1, etype=etype)
            assert len(orig_eids1) == len(orig_eids2)
            assert np.all(F.asnumpy(orig_eids1) == F.asnumpy(orig_eids2))
        parts.append(part_g)
        verify_graph_feats(hg, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids)

        shuffled_labels.append(node_feats['n1/labels'])
        shuffled_elabels.append(edge_feats['r1/labels'])
    verify_hetero_graph(hg, parts)

    shuffled_labels = F.asnumpy(F.cat(shuffled_labels, 0))
    shuffled_elabels = F.asnumpy(F.cat(shuffled_elabels, 0))
    orig_labels = np.zeros(shuffled_labels.shape, dtype=shuffled_labels.dtype)
    orig_elabels = np.zeros(shuffled_elabels.shape, dtype=shuffled_elabels.dtype)
    orig_labels[F.asnumpy(orig_nids['n1'])] = shuffled_labels
    orig_elabels[F.asnumpy(orig_eids['r1'])] = shuffled_elabels
    assert np.all(orig_labels == F.asnumpy(hg.nodes['n1'].data['labels']))
    assert np.all(orig_elabels == F.asnumpy(hg.edges['r1'].data['labels']))

def check_partition(g, part_method, reshuffle, num_parts=4, num_trainers_per_machine=1, load_feats=True):
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10), F.float32)
    g.edata['feats'] = F.tensor(np.random.randn(g.number_of_edges(), 10), F.float32)
    g.update_all(fn.copy_src('feats', 'msg'), fn.sum('msg', 'h'))
    g.update_all(fn.copy_edge('feats', 'msg'), fn.sum('msg', 'eh'))
    num_hops = 2

    orig_nids, orig_eids = partition_graph(g, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                                           part_method=part_method, reshuffle=reshuffle, return_mapping=True,
                                           num_trainers_per_machine=num_trainers_per_machine)
    part_sizes = []
    shuffled_labels = []
    shuffled_edata = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, ntypes, etypes = load_partition(
            '/tmp/partition/test.json', i, load_feats=load_feats)
        _verify_partition_data_types(part_g)
        if not load_feats:
            assert not node_feats
            assert not edge_feats
            node_feats, edge_feats = load_partition_feats('/tmp/partition/test.json', i)
        if num_trainers_per_machine > 1:
            for ntype in g.ntypes:
                name = ntype + '/trainer_id'
                assert name in node_feats
                part_ids = F.floor_div(node_feats[name], num_trainers_per_machine)
                assert np.all(F.asnumpy(part_ids) == i)

            for etype in g.etypes:
                name = etype + '/trainer_id'
                assert name in edge_feats
                part_ids = F.floor_div(edge_feats[name], num_trainers_per_machine)
                assert np.all(F.asnumpy(part_ids) == i)

        # Check the metadata
        assert gpb._num_nodes() == g.number_of_nodes()
        assert gpb._num_edges() == g.number_of_edges()

        assert gpb.num_partitions() == num_parts
        gpb_meta = gpb.metadata()
        assert len(gpb_meta) == num_parts
        assert len(gpb.partid2nids(i)) == gpb_meta[i]['num_nodes']
        assert len(gpb.partid2eids(i)) == gpb_meta[i]['num_edges']
        part_sizes.append((gpb_meta[i]['num_nodes'], gpb_meta[i]['num_edges']))

        nid = F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node'])
        local_nid = gpb.nid2localnid(nid, i)
        assert F.dtype(local_nid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
        eid = F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge'])
        local_eid = gpb.eid2localeid(eid, i)
        assert F.dtype(local_eid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_eid) == np.arange(0, len(local_eid)))

        # Check the node map.
        local_nodes = F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node'])
        llocal_nodes = F.nonzero_1d(part_g.ndata['inner_node'])
        local_nodes1 = gpb.partid2nids(i)
        assert F.dtype(local_nodes1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_nodes)) == np.sort(F.asnumpy(local_nodes1)))
        assert np.all(F.asnumpy(llocal_nodes) == np.arange(len(llocal_nodes)))

        # Check the edge map.
        local_edges = F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge'])
        llocal_edges = F.nonzero_1d(part_g.edata['inner_edge'])
        local_edges1 = gpb.partid2eids(i)
        assert F.dtype(local_edges1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_edges)) == np.sort(F.asnumpy(local_edges1)))
        assert np.all(F.asnumpy(llocal_edges) == np.arange(len(llocal_edges)))

        # Verify the mapping between the reshuffled IDs and the original IDs.
        part_src_ids, part_dst_ids = part_g.edges()
        part_src_ids = F.gather_row(part_g.ndata[dgl.NID], part_src_ids)
        part_dst_ids = F.gather_row(part_g.ndata[dgl.NID], part_dst_ids)
        part_eids = part_g.edata[dgl.EID]
        orig_src_ids = F.gather_row(orig_nids, part_src_ids)
        orig_dst_ids = F.gather_row(orig_nids, part_dst_ids)
        orig_eids1 = F.gather_row(orig_eids, part_eids)
        orig_eids2 = g.edge_ids(orig_src_ids, orig_dst_ids)
        assert F.shape(orig_eids1)[0] == F.shape(orig_eids2)[0]
        assert np.all(F.asnumpy(orig_eids1) == F.asnumpy(orig_eids2))

        if reshuffle:
            local_orig_nids = orig_nids[part_g.ndata[dgl.NID]]
            local_orig_eids = orig_eids[part_g.edata[dgl.EID]]
            part_g.ndata['feats'] = F.gather_row(g.ndata['feats'], local_orig_nids)
            part_g.edata['feats'] = F.gather_row(g.edata['feats'], local_orig_eids)
            local_nodes = orig_nids[local_nodes]
            local_edges = orig_eids[local_edges]
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
            true_feats = F.gather_row(g.ndata[name], local_nodes)
            ndata = F.gather_row(node_feats['_N/' + name], local_nid)
            assert np.all(F.asnumpy(true_feats) == F.asnumpy(ndata))
        for name in ['feats']:
            assert '_E/' + name in edge_feats
            assert edge_feats['_E/' + name].shape[0] == len(local_edges)
            true_feats = F.gather_row(g.edata[name], local_edges)
            edata = F.gather_row(edge_feats['_E/' + name], local_eid)
            assert np.all(F.asnumpy(true_feats) == F.asnumpy(edata))

        # This only works if node/edge IDs are shuffled.
        if reshuffle:
            shuffled_labels.append(node_feats['_N/labels'])
            shuffled_edata.append(edge_feats['_E/feats'])

    # Verify that we can reconstruct node/edge data for original IDs.
    if reshuffle:
        shuffled_labels = F.asnumpy(F.cat(shuffled_labels, 0))
        shuffled_edata = F.asnumpy(F.cat(shuffled_edata, 0))
        orig_labels = np.zeros(shuffled_labels.shape, dtype=shuffled_labels.dtype)
        orig_edata = np.zeros(shuffled_edata.shape, dtype=shuffled_edata.dtype)
        orig_labels[F.asnumpy(orig_nids)] = shuffled_labels
        orig_edata[F.asnumpy(orig_eids)] = shuffled_edata
        assert np.all(orig_labels == F.asnumpy(g.ndata['labels']))
        assert np.all(orig_edata == F.asnumpy(g.edata['feats']))

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

def check_hetero_partition_single_etype(num_trainers):
    user_ids = np.arange(1000)
    item_ids = np.arange(2000)
    num_edges = 3 * 1000
    src_ids = np.random.choice(user_ids, size=num_edges)
    dst_ids = np.random.choice(item_ids, size=num_edges)
    hg = dgl.heterograph({('user', 'like', 'item'): (src_ids, dst_ids)})

    with tempfile.TemporaryDirectory() as test_dir:
        orig_nids, orig_eids = partition_graph(
            hg, 'test', 2, test_dir, num_trainers_per_machine=num_trainers, return_mapping=True)
        assert len(orig_nids) == len(hg.ntypes)
        assert len(orig_eids) == len(hg.etypes)
        for ntype in hg.ntypes:
            assert len(orig_nids[ntype]) == hg.number_of_nodes(ntype)
        for etype in hg.etypes:
            assert len(orig_eids[etype]) == hg.number_of_edges(etype)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_partition():
    os.environ['DGL_DIST_DEBUG'] = '1'
    g = create_random_graph(1000)
    check_partition(g, 'metis', False)
    check_partition(g, 'metis', True)
    check_partition(g, 'metis', True, 4, 8)
    check_partition(g, 'metis', True, 1, 8)
    check_partition(g, 'random', False)
    check_partition(g, 'random', True)
    check_partition(g, 'metis', True, 4, 8, load_feats=False)
    reset_envs()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
def test_hetero_partition():
    os.environ['DGL_DIST_DEBUG'] = '1'
    check_hetero_partition_single_etype(1)
    check_hetero_partition_single_etype(4)
    hg = create_random_hetero()
    check_hetero_partition(hg, 'metis')
    check_hetero_partition(hg, 'metis', 1, 8)
    check_hetero_partition(hg, 'metis', 4, 8)
    check_hetero_partition(hg, 'random')
    check_hetero_partition(hg, 'metis', 4, 8, load_feats=False)
    reset_envs()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_BasicPartitionBook():
    part_id = 0
    num_parts = 2
    node_map = np.random.choice(num_parts, 1000)
    edge_map = np.random.choice(num_parts, 5000)
    graph = dgl.rand_graph(1000, 5000)
    graph = dgl.node_subgraph(graph, F.arange(0, graph.num_nodes()))
    gpb = BasicPartitionBook(part_id, num_parts, node_map, edge_map, graph)
    c_etype = ('_N', '_E', '_N')
    assert gpb.etypes == ['_E']
    assert gpb.canonical_etypes == [c_etype]

    node_policy = NodePartitionPolicy(gpb, '_N')
    assert node_policy.type_name == '_N'
    edge_policy = EdgePartitionPolicy(gpb, '_E')
    assert edge_policy.type_name == '_E'

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_RangePartitionBook():
    part_id = 0
    num_parts = 2
    # homogeneous
    node_map = {'_N': F.tensor([[0, 1000], [1000, 2000]])}
    edge_map = {'_E': F.tensor([[0, 5000], [5000, 10000]])}
    ntypes = {'_N': 0}
    etypes = {'_E': 0}
    gpb = RangePartitionBook(
        part_id, num_parts, node_map, edge_map, ntypes, etypes)
    assert gpb.etypes == ['_E']
    assert gpb.canonical_etypes == [None]
    assert gpb._to_canonical_etype('_E') == '_E'

    node_policy = NodePartitionPolicy(gpb, '_N')
    assert node_policy.type_name == '_N'
    edge_policy = EdgePartitionPolicy(gpb, '_E')
    assert edge_policy.type_name == '_E'

    # heterogeneous, init via etype
    node_map = {'node1': F.tensor([[0, 1000], [1000, 2000]]), 'node2': F.tensor([
        [0, 1000], [1000, 2000]])}
    edge_map = {'edge1': F.tensor([[0, 5000], [5000, 10000]])}
    ntypes = {'node1': 0, 'node2': 1}
    etypes = {'edge1': 0}
    gpb = RangePartitionBook(
        part_id, num_parts, node_map, edge_map, ntypes, etypes)
    assert gpb.etypes == ['edge1']
    assert gpb.canonical_etypes == [None]
    assert gpb._to_canonical_etype('edge1') == 'edge1'

    node_policy = NodePartitionPolicy(gpb, 'node1')
    assert node_policy.type_name == 'node1'
    edge_policy = EdgePartitionPolicy(gpb, 'edge1')
    assert edge_policy.type_name == 'edge1'

    # heterogeneous, init via canonical etype
    node_map = {'node1': F.tensor([[0, 1000], [1000, 2000]]), 'node2': F.tensor([
        [0, 1000], [1000, 2000]])}
    edge_map = {('node1', 'edge1', 'node2'): F.tensor([[0, 5000], [5000, 10000]])}
    ntypes = {'node1': 0, 'node2': 1}
    etypes = {('node1', 'edge1', 'node2'): 0}
    c_etype = list(etypes.keys())[0]
    gpb = RangePartitionBook(
        part_id, num_parts, node_map, edge_map, ntypes, etypes)
    assert gpb.etypes == ['edge1']
    assert gpb.canonical_etypes == [c_etype]
    assert gpb._to_canonical_etype('edge1') == c_etype
    assert gpb._to_canonical_etype(c_etype) == c_etype
    expect_except = False
    try:
        gpb._to_canonical_etype(('node1', 'edge2', 'node2'))
    except:
        expect_except = True
    assert expect_except
    expect_except = False
    try:
        gpb._to_canonical_etype('edge2')
    except:
        expect_except = True
    assert expect_except

    node_policy = NodePartitionPolicy(gpb, 'node1')
    assert node_policy.type_name == 'node1'
    edge_policy = EdgePartitionPolicy(gpb, c_etype)
    assert edge_policy.type_name == c_etype

    data_name = HeteroDataName(False, 'edge1', 'edge1')
    assert data_name.get_type() == 'edge1'
    data_name = HeteroDataName(False, c_etype, 'edge1')
    assert data_name.get_type() == c_etype

if __name__ == '__main__':
    os.makedirs('/tmp/partition', exist_ok=True)
    test_partition()
    test_hetero_partition()
    test_BasicPartitionBook()
    test_RangePartitionBook()
