import dgl
import dgl.graphbolt
import torch
import torchdata.datapipes as dp


def test_SubgraphSampler():
    graph = dgl.add_reverse_edges(dgl.rand_graph(20, 60))
    sampler = dgl.dataloading.NeighborSampler([2, 2])

    def sampler_func(data):
        return sampler.sample(graph, data)

    minibatch_dp = dp.map.SequenceWrapper(torch.arange(10)).batch(2)
    sampler_dp = dgl.graphbolt.SubgraphSampler(minibatch_dp, sampler_func)

    assert len(list(sampler_dp)) == 5
