"""Uniform negative sampler for GraphBolt."""

from ..subgraph_sampler import SubgraphSampler
import torch

class NeighborSampler(SubgraphSampler):
    def __init__(
    self,
    datapipe,
    input_format,
    graph,
    fanouts,
    replace=False,
    prob_name=None,
    ):
        """
        Initlization for a link neighbor subgraph sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        input_format : LinkPredictionEdgeFormat
            Determines the format of the input data.
        graph : CSCSamplingGraph
            The graph on which to perform subgraph sampling.
        fanouts: list[list[int]]
            The number of edges to be sampled for each node with or without
            considering edge types.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        prob_name: str, optional
            The name of an edge attribute used a. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.

        Examples
        --------
        >>> from dgl import graphbolt as gb
        >>> indptr = torch.LongTensor([0, 2, 4, 5])
        >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
        >>> graph = gb.from_csc(indptr, indices)
        >>> data_format = gb.LinkPredictionEdgeFormat.INDEPENDENT
        >>> node_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        >>> item_set = gb.ItemSet(node_pairs)
        >>> minibatch_sampler = gb.MinibatchSampler(
            ...item_set, batch_size=1,
            ...)
        >>> neg_sampler = gb.UniformNegativeSampler(
            ...minibatch_sampler, 2, data_format, graph)
        >>> fanouts = [[5], [10], [15]]
        >>> subgraph_sampler = gb.LinkNeighborSampler(
            ...neg_sampler, data_format, fanouts)
        >>> for data in subgraph_sampler:
            ...  print(data)
            ...
        (tensor([0, 0, 0]), tensor([1, 1, 2]), tensor([1, 0, 0]))
        (tensor([1, 1, 1]), tensor([2, 1, 2]), tensor([1, 0, 0]))
        """
        super().__init__(datapipe, fanouts, replace, prob_name)
        self.graph = graph

    def _sample_sub_graph(self, seeds, hop):                
        return self.graph.sample_neighbors(
            seeds,
            torch.LongTensor(self.fanouts[hop]),
            self.replace,
            self.prob_name,
        )
