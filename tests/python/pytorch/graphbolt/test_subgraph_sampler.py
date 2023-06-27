import dgl
import dgl.graphbolt
import gb_test_utils
import torch
import torchdata.datapipes as dp


def test_SubgraphSampler():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)

    def sampler_func(data):
        adjs = []
        seeds = data

        for hop in range(2):
            sg = graph.sample_neighbors(seeds, torch.LongTensor([2]))
            seeds = sg.indices
            adjs.insert(0, sg)
        return seeds, data, adjs

    minibatch_dp = dp.map.SequenceWrapper(torch.arange(10)).batch(2)
    minibatch_dp = minibatch_dp.to_iter_datapipe().collate()
    sampler_dp = dgl.graphbolt.SubgraphSampler(minibatch_dp, sampler_func)

    assert len(list(sampler_dp)) == 5
