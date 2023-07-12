import multiprocessing as mp
import unittest

import backend as F

import dgl
import dgl.graphbolt as gb

import torch


def subprocess_entry(queue, barrier):
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

    # Send the data twice (back and forth) and then verify.
    # Method get() and put() of mp.Queue is blocking by default.
    # Step 1. Put the data.
    queue.put(adjs)
    # Step 2. Another process gets the data.
    # Step 3. Barrier. Wait for another process to get the data.
    barrier.wait()
    # Step 4. Another process puts the data.
    # Step 5. Get the data.
    result = queue.get()
    # Step 6. Verification.
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
    queue = mp.Queue()
    barrier = mp.Barrier(2)
    proc = mp.Process(target=subprocess_entry, args=(queue, barrier))
    proc.start()

    # Send the data twice (back and forth) and then verify.
    # Method get() and put() of mp.Queue is blocking by default.
    # Step 1. Another process puts the data.
    # Step 2. Get the data. This operation will block if the queue is empty.
    items = queue.get()
    # Step 3. Barrier.
    barrier.wait()
    # Step 4. Put the data again.
    queue.put(items)
    # Step 5. Another process gets the final data.
    # Step 6. Wait for another process to end
    proc.join()
