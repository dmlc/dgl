import dgl
import dgl.graphbolt
import gb_test_utils
import pytest
import torch
import torchdata.datapipes as dp


def get_graphbolt_sampler_func():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)

    def sampler_func(data):
        adjs = []
        seeds = data

        for hop in range(2):
            sg = graph.sample_neighbors(seeds, torch.LongTensor([2]))
            seeds = sg.node_pairs[0]
            adjs.insert(0, sg)
        return seeds, data, adjs

    return sampler_func


def get_dgl_sampler_func():
    graph = dgl.add_reverse_edges(dgl.rand_graph(20, 60))
    sampler = dgl.dataloading.NeighborSampler([2, 2])

    def sampler_func(data):
        return sampler.sample(graph, data)

    return sampler_func


def get_graphbolt_minibatch_dp():
    itemset = dgl.graphbolt.ItemSet(torch.arange(10))
    return dgl.graphbolt.MinibatchSampler(itemset, batch_size=2)


def get_torchdata_minibatch_dp():
    minibatch_dp = dp.map.SequenceWrapper(torch.arange(10)).batch(2)
    minibatch_dp = minibatch_dp.to_iter_datapipe().collate()
    return minibatch_dp


@pytest.mark.parametrize(
    "sampler_func", [get_graphbolt_sampler_func(), get_dgl_sampler_func()]
)
@pytest.mark.parametrize(
    "minibatch_dp", [get_graphbolt_minibatch_dp(), get_torchdata_minibatch_dp()]
)
def test_SubgraphSampler(minibatch_dp, sampler_func):
    sampler_dp = dgl.graphbolt.SubgraphSampler(minibatch_dp, sampler_func)
    assert len(list(sampler_dp)) == 5
