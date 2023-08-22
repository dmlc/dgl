"""Neighbor subgraph sampler for GraphBolt."""

from ..subgraph_sampler import SubgraphSampler


class NeighborSampler(SubgraphSampler):
    """
    Neighbor sampler is responsible for sampling a subgraph from given data. It
    returns an induced subgraph along with compacted information. In the
    context of a node classification task, the neighbor sampler directly
    utilizes the nodes provided as seed nodes. However, in scenarios involving
    link prediction, the process needs another pre-peocess operation. That is,
    gathering unique nodes from the given node pairs, encompassing both
    positive and negative node pairs, and employs these nodes as the seed nodes
    for subsequent steps.
    """
    
    def __init__(
        self,
        datapipe,
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
        graph : CSCSamplingGraph
            The graph on which to perform subgraph sampling.
        fanouts: list[torch.Tensor]
            The number of edges to be sampled for each node with or without
            considering edge types. The length of this parameter implicitly
            signifies the layer of sampling being conducted.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        prob_name: str, optional
            The name of an edge attribute used as the weights of sampling for
            each node. This attribute tensor should contain (unnormalized)
            probabilities corresponding to each neighboring edge of a node.
            It must be a 1D floating-point or boolean tensor, with the number
            of elements equalling the total number of edges.

        Examples
        -------
        >>> import dgl.graphbolt as gb
        >>> from torchdata.datapipes.iter import Mapper
        >>> def to_link_block(data):
            ... block = gb.LinkPredictionBlock(node_pair=data)
            ... return block
            ...
        >>> from dgl import graphbolt as gb
        >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
        >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
        >>> graph = gb.from_csc(indptr, indices)
        >>> data_format = gb.LinkPredictionEdgeFormat.INDEPENDENT
        >>> node_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        >>> item_set = gb.ItemSet(node_pairs)
        >>> minibatch_sampler = gb.MinibatchSampler(
            ...item_set, batch_size=1,
            ...)
        >>> data_block_converter = Mapper(minibatch_sampler, to_link_block)
        >>> neg_sampler = gb.UniformNegativeSampler(
            ...data_block_converter, 2, data_format, graph)
        >>> fanouts = [torch.LongTensor([5]), torch.LongTensor([10]),
            ...torch.LongTensor([15])]
        >>> subgraph_sampler = gb.NeighborSampler(
            ...neg_sampler, graph, fanouts)
        >>> for data in subgraph_sampler:
            ... print(data.compacted_node_pair)
            ... print(len(data.sampled_subgraphs))
        (tensor([0, 0, 0]), tensor([1, 0, 2]))
        3
        (tensor([0, 0, 0]), tensor([1, 1, 1]))
        3
        """
        super().__init__(datapipe, fanouts, replace, prob_name)
        self.graph = graph

    def _sample_sub_graph(self, seeds, hop):
        return self.graph.sample_neighbors(
            seeds,
            self.fanouts[hop],
            self.replace,
            self.prob_name,
        )
