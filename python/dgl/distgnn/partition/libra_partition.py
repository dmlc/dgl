r"""Libra partition functions.

Libra partition is a vertex-cut based partitioning algorithm from
`Distributed Power-law Graph Computing:
Theoretical and Empirical Analysis
<https://proceedings.neurips.cc/paper/2014/file/67d16d00201083a2b118dd5128dd6f59-Paper.pdf>`__
from Xie et al.
"""

# Copyright (c) 2021 Intel Corporation
#  \file distgnn/partition/libra_partition.py
#  \brief Libra - Vertex-cut based graph partitioner for distributed training
#  \author Vasimuddin Md <vasimuddin.md@intel.com>,
#          Guixiang Ma <guixiang.ma@intel.com>
#          Sanchit Misra <sanchit.misra@intel.com>,
#          Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
#          Sasikanth Avancha <sasikanth.avancha@intel.com>
#          Nesreen K. Ahmed <nesreen.k.ahmed@intel.com>
#  \cite Distributed Power-law Graph Computing: Theoretical and Empirical Analysis

import json
import os
import time

import torch as th

from dgl import DGLGraph
from dgl._sparse_ops import (
    libra2dgl_build_adjlist,
    libra2dgl_build_dict,
    libra2dgl_set_lr,
    libra_vertex_cut,
)
from dgl.base import DGLError
from dgl.data.utils import save_graphs, save_tensors


def libra_partition(num_community, G, resultdir):
    """
    Performs vertex-cut based graph partitioning and converts the partitioning
    output to DGL input format.

    Parameters
    ----------
    num_community : Number of partitions to create
    G : Input graph to be partitioned
    resultdir : Output location for storing the partitioned graphs

    Output
    ------
    1. Creates X partition folder as XCommunities (say, X=2, so, 2Communities)
       XCommunities contains file name communityZ.txt per partition Z (Z <- 0 .. X-1);
       each such file contains a list of edges assigned to that partition.
       These files constitute the output of Libra graph partitioner
       (An intermediate result of this function).
    2. The folder also contains partZ folders, each of these folders stores
       DGL/DistGNN graphs for the Z partitions;
       these graph files are used as input to DistGNN.
    3. The folder also contains a json file which contains partitions' information.
    """

    num_nodes = G.num_nodes()  # number of nodes
    num_edges = G.num_edges()  # number of edges
    print("Number of nodes in the graph: ", num_nodes)
    print("Number of edges in the graph: ", num_edges)

    in_d = G.in_degrees()
    out_d = G.out_degrees()
    node_degree = in_d + out_d
    edgenum_unassigned = node_degree.clone()

    u_t, v_t = G.edges()
    weight_ = th.ones(u_t.shape[0], dtype=th.int64)
    community_weights = th.zeros(num_community, dtype=th.int64)

    # self_loop = 0
    # for p, q in zip(u_t, v_t):
    #     if p == q:
    #         self_loop += 1
    # print("#self loops in the dataset: ", self_loop)

    # del G

    ## call to C/C++ code
    out = th.zeros(u_t.shape[0], dtype=th.int32)
    libra_vertex_cut(
        num_community,
        node_degree,
        edgenum_unassigned,
        community_weights,
        u_t,
        v_t,
        weight_,
        out,
        num_nodes,
        num_edges,
        resultdir,
    )

    print("Max partition size: ", int(community_weights.max()))
    print(" ** Converting libra partitions to dgl graphs **")
    fsize = int(community_weights.max()) + 1024  ## max edges in partition
    # print("fsize: ", fsize, flush=True)

    node_map = th.zeros(num_community, dtype=th.int64)
    indices = th.zeros(num_nodes, dtype=th.int64)
    lrtensor = th.zeros(num_nodes, dtype=th.int64)
    gdt_key = th.zeros(num_nodes, dtype=th.int64)
    gdt_value = th.zeros([num_nodes, num_community], dtype=th.int64)
    offset = th.zeros(1, dtype=th.int64)
    ldt_ar = []

    gg_ar = [DGLGraph() for i in range(num_community)]
    part_nodes = []

    print(">>> ", "num_nodes   ", " ", "num_edges")
    ## Iterator over number of partitions
    for i in range(num_community):
        g = gg_ar[i]

        a_t = th.zeros(fsize, dtype=th.int64)
        b_t = th.zeros(fsize, dtype=th.int64)
        ldt_key = th.zeros(fsize, dtype=th.int64)
        ldt_ar.append(ldt_key)

        ## building node, parition dictionary
        ## Assign local node ids and mapping to global node ids
        ret = libra2dgl_build_dict(
            a_t,
            b_t,
            indices,
            ldt_key,
            gdt_key,
            gdt_value,
            node_map,
            offset,
            num_community,
            i,
            fsize,
            resultdir,
        )

        num_nodes_partition = int(ret[0])
        num_edges_partition = int(ret[1])
        part_nodes.append(num_nodes_partition)
        print(">>> ", num_nodes_partition, " ", num_edges_partition)
        g.add_edges(a_t[0:num_edges_partition], b_t[0:num_edges_partition])

    ########################################################
    ## fixing lr - 1-level tree for the split-nodes
    libra2dgl_set_lr(gdt_key, gdt_value, lrtensor, num_community, num_nodes)
    ########################################################
    # graph_name = dataset
    graph_name = resultdir.split("_")[-1].split("/")[0]
    part_method = "Libra"
    num_parts = num_community  ## number of paritions/communities
    num_hops = 0
    node_map_val = node_map.tolist()
    edge_map_val = 0
    out_path = resultdir

    part_metadata = {
        "graph_name": graph_name,
        "num_nodes": G.num_nodes(),
        "num_edges": G.num_edges(),
        "part_method": part_method,
        "num_parts": num_parts,
        "halo_hops": num_hops,
        "node_map": node_map_val,
        "edge_map": edge_map_val,
    }
    ############################################################

    for i in range(num_community):
        g = gg_ar[0]
        num_nodes_partition = part_nodes[i]
        adj = th.zeros([num_nodes_partition, num_community - 1], dtype=th.int64)
        inner_node = th.zeros(num_nodes_partition, dtype=th.int32)
        lr_t = th.zeros(num_nodes_partition, dtype=th.int64)
        ldt = ldt_ar[0]

        try:
            feat = G.ndata["feat"]
        except KeyError:
            feat = G.ndata["features"]

        try:
            labels = G.ndata["label"]
        except KeyError:
            labels = G.ndata["labels"]

        trainm = G.ndata["train_mask"].int()
        testm = G.ndata["test_mask"].int()
        valm = G.ndata["val_mask"].int()

        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes_partition, feat_size], dtype=feat.dtype)

        glabels = th.zeros(num_nodes_partition, dtype=labels.dtype)
        gtrainm = th.zeros(num_nodes_partition, dtype=trainm.dtype)
        gtestm = th.zeros(num_nodes_partition, dtype=testm.dtype)
        gvalm = th.zeros(num_nodes_partition, dtype=valm.dtype)

        ## build remote node databse per local node
        ## gather feats, train, test, val, and labels for each partition
        libra2dgl_build_adjlist(
            feat,
            gfeat,
            adj,
            inner_node,
            ldt,
            gdt_key,
            gdt_value,
            node_map,
            lr_t,
            lrtensor,
            num_nodes_partition,
            num_community,
            i,
            feat_size,
            labels,
            trainm,
            testm,
            valm,
            glabels,
            gtrainm,
            gtestm,
            gvalm,
            feat.shape[0],
        )

        g.ndata["adj"] = adj  ## database of remote clones
        g.ndata["inner_node"] = inner_node  ## split node '0' else '1'
        g.ndata["feat"] = gfeat  ## gathered features
        g.ndata["lf"] = lr_t  ## 1-level tree among split nodes

        g.ndata["label"] = glabels
        g.ndata["train_mask"] = gtrainm
        g.ndata["test_mask"] = gtestm
        g.ndata["val_mask"] = gvalm

        # Validation code, run only small graphs
        # for l in range(num_nodes_partition):
        #     index = int(ldt[l])
        #     assert glabels[l] == labels[index]
        #     assert gtrainm[l] == trainm[index]
        #     assert gtestm[l] == testm[index]
        #     for j in range(feat_size):
        #         assert gfeat[l][j] == feat[index][j]

        print("Writing partition {} to file".format(i), flush=True)

        part = g
        part_id = i
        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata["part-{}".format(part_id)] = {
            "node_feats": node_feat_file,
            "edge_feats": edge_feat_file,
            "part_graph": part_graph_file,
        }
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, part.ndata)
        save_graphs(part_graph_file, [part])

        del g
        del gg_ar[0]
        del ldt
        del ldt_ar[0]

    with open("{}/{}.json".format(out_path, graph_name), "w") as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

    print("Conversion libra2dgl completed !!!")


def partition_graph(num_community, G, resultdir):
    """
    Performs vertex-cut based graph partitioning and converts the partitioning
    output to DGL input format.

    Given a graph, this function will create a folder named ``XCommunities`` where ``X``
    stands for the number of communities.  It will contain ``X`` files named
    ``communityZ.txt`` for each partition Z (from 0 to X-1);
    each such file contains a list of edges assigned to that partition.
    These files constitute the output of Libra graph partitioner.

    The folder also contains X subfolders named ``partZ``, each of these folders stores
    DGL/DistGNN graphs for partition Z; these graph files are used as input to
    DistGNN.

    The folder also contains a json file which contains partitions' information.

    Currently we require the graph's node data to contain the following columns:

    * ``features`` for node features.
    * ``label`` for node labels.
    * ``train_mask`` as a boolean mask of training node set.
    * ``val_mask`` as a boolean mask of validation node set.
    * ``test_mask`` as a boolean mask of test node set.

    Parameters
    ----------
    num_community : int
        Number of partitions to create.
    G : DGLGraph
        Input graph to be partitioned.
    resultdir : str
        Output location for storing the partitioned graphs.
    """

    print("num partitions: ", num_community)
    print("output location: ", resultdir)

    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        raise DGLError("Error: Could not create directory: ", resultdir)

    tic = time.time()
    print(
        "####################################################################"
    )
    print("Executing parititons: ", num_community)
    ltic = time.time()
    try:
        resultdir = os.path.join(resultdir, str(num_community) + "Communities")
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        raise DGLError("Error: Could not create sub-directory: ", resultdir)

    ## Libra partitioning
    libra_partition(num_community, G, resultdir)

    ltoc = time.time()
    print(
        "Time taken by {} partitions {:0.4f} sec".format(
            num_community, ltoc - ltic
        )
    )
    print()

    toc = time.time()
    print(
        "Generated ",
        num_community,
        " partitions in {:0.4f} sec".format(toc - tic),
        flush=True,
    )
    print("Partitioning completed successfully !!!")
