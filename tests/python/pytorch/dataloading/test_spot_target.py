from collections.abc import Mapping

import dgl
import numpy as np
import pytest
import torch


def _create_homogeneous():
    s = torch.randint(0, 200, (1000,))
    d = torch.randint(0, 200, (1000,))
    g = dgl.graph((s, d), num_nodes=200)
    reverse_eids = torch.cat([torch.arange(1000, 2000), torch.arange(0, 1000)])
    seed_edges = torch.arange(0, 1000)
    return g, reverse_eids, seed_edges


def _find_edges_to_exclude(g, pair_eids, degree_threshold):
    src, dst = g.find_edges(pair_eids)
    head_degree = g.in_degrees(src)
    tail_degree = g.in_degrees(dst)
    degree = torch.min(head_degree, tail_degree)
    degree_mask = degree < degree_threshold
    low_degree_pair_eids = pair_eids[degree_mask]
    low_degree_pair_eids = torch.cat(
        [low_degree_pair_eids, low_degree_pair_eids + 1000]
    )
    return low_degree_pair_eids


@pytest.mark.parametrize("degree_threshold", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batch_size", [1, 10, 50])
def test_spot_target_excludes(degree_threshold, batch_size):
    g, reverse_eids, seed_edges = _create_homogeneous()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    low_degree_excluder = dgl.dataloading.SpotTarget(
        g,
        exclude="reverse_id",
        degree_threshold=degree_threshold,
        reverse_eids=reverse_eids,
    )
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude=low_degree_excluder,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
    )
    dataloader = dgl.dataloading.DataLoader(
        g, seed_edges, sampler, batch_size=batch_size
    )

    for i, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
        dataloader
    ):
        if isinstance(blocks, list):
            subg = blocks[0]
        else:
            subg = blocks
        pair_eids = pair_graph.edata[dgl.EID]
        block_eids = subg.edata[dgl.EID]
        edges_to_exclude = _find_edges_to_exclude(
            g, pair_eids, degree_threshold
        )
        if edges_to_exclude is None:
            continue
        edges_to_exclude = dgl.utils.recursive_apply(
            edges_to_exclude, lambda x: x.cpu().numpy()
        )
        block_eids = dgl.utils.recursive_apply(
            block_eids, lambda x: x.cpu().numpy()
        )

        if isinstance(edges_to_exclude, Mapping):
            for k in edges_to_exclude.keys():
                assert not np.isin(edges_to_exclude[k], block_eids[k]).any()
        else:
            assert not np.isin(edges_to_exclude, block_eids).any()

        if i == 10:
            break


if __name__ == "__main__":
    test_spot_target_excludes(degree_threshold=2, batch_size=10)
