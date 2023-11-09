import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch

def test_InSubgraphSampler_invoke():
    # Instantiate graph and required datapipes.
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, 2 * num_seeds).reshape(-1, 2), names="node_pairs"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)
    # in_subgraph_sampler = gb.Ne