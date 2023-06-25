"""Subgraph samplers"""

from torchdata.datapipes.iter import IterDataPipe


class MockSubgraphSampler(IterDataPipe):
    """A mock subgraph sampler."""

    def __init__(self, dp, graph, sampler):
        super().__init__()
        self.graph = graph
        self.dp = dp
        self.sampler = sampler

    def __iter__(self):
        for data in self.dp:
            sg = self.sampler.sample(self.graph, data)
            yield sg
