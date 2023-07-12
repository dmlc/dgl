import multiprocessing as mp
import os
import tempfile
import time
import unittest

import backend as F

import dgl
import dgl.graphbolt as gb
import gb_test_utils as gbt

import pytest
import torch

def subprocess_entry(q):
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    type_per_edge = torch.LongTensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)
    ntypes = {"n1": 0, "n2": 1, "n3": 2}
    etypes = {("n1", "e1", "n2"): 0, ("n1", "e2", "n3"): 1}
    metadata = gb.GraphMetadata(ntypes, etypes)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(
        indptr, indices, type_per_edge=type_per_edge, metadata=metadata
    )

    adjs = []
    seeds = torch.arange(5)

    # Sampling.
    for hop in range(2):
        sg = graph.sample_neighbors(seeds, torch.LongTensor([2]))
        seeds = sg.indices
        adjs.append(sg)

    # 1. Put.
    q.put(adjs)
    time.sleep(2)
    # 4. Get.
    result = q.get()

    # Verification.
    for hop in range(2):
        # Tensors.
        assert torch.equal(adjs[hop].indptr, result[hop].indptr)
        assert torch.equal(adjs[hop].indices, result[hop].indices)
        assert torch.equal(
            adjs[hop].reverse_column_node_ids,
            result[hop].reverse_column_node_ids,
        )
        # Optional tensors.
        assert (
            adjs[hop].reverse_row_node_ids is None
            and adjs[hop].reverse_row_node_ids is None
        ) or torch.equal(
            adjs[hop].reverse_row_node_ids, result[hop].reverse_row_node_ids
        )
        assert (
            adjs[hop].reverse_edge_ids is None
            and result[hop].reverse_edge_ids is None
        ) or torch.equal(
            adjs[hop].reverse_edge_ids, result[hop].reverse_edge_ids
        )
        assert (
            adjs[hop].type_per_edge is None
            and result[hop].type_per_edge is None
        ) or torch.equal(adjs[hop].type_per_edge, result[hop].type_per_edge)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_subgraph_serialization():
    # Create a sub-process.
    q = mp.Queue()
    proc = mp.Process(target=subprocess_entry, args=(q,))
    proc.start()

    # 2. Get.
    items = q.get()
    # 3. Put again.
    q.put(items)

    time.sleep(1)
    proc.join()
