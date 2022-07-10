import os
import json
import numpy as np
import dgl
import torch as th

def verify_ci_partitions(part_graph, node_feats, edge_feats, metadata, rank, world_size): 
    '''
    A simple verification function used to perform sanity check on the results generated
    by the dist. partitioning pipeline for a sample CI Test Graph. 

    CI Graph, as described in the constants.py is generated and dgl objects are generated
    using the usual dist. pipeline functionality. 

    If the ci_mode parameter is passed in the command line while launching the dist. partitioning
    pipeline then this pipeline operates in the CI_MODE and it does the following: 
    1. node-id to partition-id mappings are generated randomly. However the seed is set to `0`.
       This ensure that same node-id to partition-id mappings are used by both the ranks. 
    2. dgl objects and metadata json objects are created, but are not written into files. 
    3. This function, which is executed only on rank-0, takes the metadata and partitioned graph
       object generated on rank-0 and runs following sanity checks on the this graph. 
	a. node types in the partitioned graph match with the original graph
	b. edge types in the partitioned graph match with the original graph
	c. node features in the partitioned graph match with the original graph. 

    Parameters:
    -----------
    dgl graph object : graph object
        dgl graph object, which is created as per the plan and revisited in case others feel
        this as out-of-the-line appearance

    node_feats : tensor
        a tensor for the node features consolidated on this rank after the datashuffle

    edge_feats : tensor
        a tensor for the edge features consolidated on this rank after the datashuffle

    metadata : json object
        this metadata is a consolidated metadata after the data shuffle request

    rank : integer
        a unique number assigned to the current machine

    world_size : integer
        integer indicating total no. of processes in the experiment setup

    '''

    node_features = node_feats
    edge_features = edge_feats

    ntype_map = metadata['ntypes']
    ntypes = [None] * len(ntype_map)
    for k in ntype_map:
        ntype_id = ntype_map[k]
        ntypes[ntype_id] = k

    etype_map = metadata['etypes']
    etypes = [None] * len(etype_map)
    for k in etype_map:
        etype_id = etype_map[k]
        etypes[etype_id] = k

    node_map = metadata['node_map']
    for key in node_map:
        node_map[key] = th.stack([th.tensor(row) for row in node_map[key]], 0)
    nid_map = dgl.distributed.id_map.IdMap(node_map)

    edge_map = metadata['edge_map']
    for key in edge_map:
        edge_map[key] = th.stack([th.tensor(row) for row in edge_map[key]], 0)
    eid_map = dgl.distributed.id_map.IdMap(edge_map)

    subg = part_graph

    subg_src_id, subg_dst_id = subg.edges()
    orig_src_id = subg.ndata['orig_id'][subg_src_id]
    orig_dst_id = subg.ndata['orig_id'][subg_dst_id]
    global_src_id = subg.ndata[dgl.NID][subg_src_id]
    global_dst_id = subg.ndata[dgl.NID][subg_dst_id]

    subg_ntype = subg.ndata[dgl.NTYPE]
    subg_etype = subg.edata[dgl.ETYPE]

    for ntype_id in th.unique(subg_ntype):
        ntype = ntypes[ntype_id]
        idx = subg_ntype == ntype_id

        nid = subg.ndata[dgl.NID][idx == 1]
        ntype_ids1, type_nid = nid_map(nid)

        orig_type_nid = subg.ndata['orig_id'][idx == 1]
        inner_node = subg.ndata['inner_node'][idx == 1]

        assert np.all(ntype_ids1.numpy() == int(ntype_id))
        assert np.all(nid[inner_node == 1].numpy() == np.arange(node_map[ntype][rank,0], node_map[ntype][rank,1]))

        #node feature test
        if(ntype_id == 0):
            orig_type_nid = orig_type_nid[inner_node == 1]
            type_nid = type_nid[inner_node == 1]

            c = (orig_type_nid >= 9)
            r0_orig_type_nids = orig_type_nid[c == 0]
            r1_orig_type_nids = orig_type_nid[c == 1]

            r0_type_nid = type_nid[c == 0]
            r1_type_nid = type_nid[c == 1]

            r0feats = node_feats[ntype+'/nodes-feat-01'][r0_type_nid]
            r1feats = node_feats[ntype+'/nodes-feat-01'][r1_type_nid]

            assert np.all(r0feats.numpy() == np.ones(r0feats.numpy().shape))
            assert np.all(r1feats.numpy() == np.ones(r1feats.numpy().shape)*2)

            #feature 2 data -- nodes-feat-02
            r0feats = node_feats[ntype+'/nodes-feat-02'][r0_type_nid]
            r1feats = node_feats[ntype+'/nodes-feat-02'][r1_type_nid]
            assert np.all(r0feats.numpy() == np.ones(r0feats.numpy().shape)*10)
            assert np.all(r1feats.numpy() == np.ones(r1feats.numpy().shape)*2*10)


    for etype_id in th.unique(subg_etype):
        etype = etypes[etype_id]
        idx = subg_etype == etype_id

        ntype_ids, type_nid = nid_map(global_src_id[idx])
        assert len(th.unique(ntype_ids)) == 1

        ntype_ids, type_nid = nid_map(global_dst_id[idx])
        assert len(th.unique(ntype_ids)) == 1

        # This is global IDs after reshuffle.
        eid = subg.edata[dgl.EID][idx]
        etype_ids1, type_eid = eid_map(eid)
        orig_type_eid = subg.edata['orig_id'][idx]
        inner_edge = subg.edata['inner_edge'][idx]
        assert np.all(etype_ids1.numpy() == int(etype_id))
        assert np.all(np.sort(eid[inner_edge == 1].numpy()) == np.arange(
            edge_map[etype][rank, 0], edge_map[etype][rank, 1]))

