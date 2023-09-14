"""Subgraph transformers."""
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import Mapper


@functional_datapipe("transform_subgraphs")
class SubgraphTransformer(Mapper):
    """A subgraph transformer used to convert subgraphs into different
    structures."""

    def __init__(
        self,
        datapipe,
        fn,
    ):
        """
        Initlization for a subgraph transformer.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        fn:
            The function applied to each minibatch which is responsible for
            converting subgraph structures, potentially utilizing other fields
            within the minibatch as arguments.

        Examples
        --------
        >>> from dgl import graphbolt as gb
        >>> def exclude_seed_edges(minibatch):
            ...edges_to_exclude = gb.add_reverse_edges(minibatch.node_pairs)
            ...minibatch.sampled_subgraphs = [
            ...subgraph.exclude_edges(edges_to_exclude)
            ...for subgraph in minibatch.sampled_subgraphs
            ...]
            ...return minibatch
        >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
        >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
        >>> graph = gb.from_csc(indptr, indices)
        >>> node_pairs = torch.LongTensor([[0, 1], [1, 2]])
        >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
        >>> item_sampler = gb.ItemSampler(
            ...item_set, batch_size=2,
            ...)
        >>> fanouts = [5, 10]
        >>> subgraph_sampler = gb.NeighborSampler(
            ...item_sampler, graph, fanouts)
        >>> subgraph_transformer = gb.SubgraphTransformer(subgraph_sampler,
            ...exclude_seed_edges)
        >>> for minibatch in subgraph_transformer:
                print minibatch.node_pairs
                for subgraph in minibatch.sampled_subgraphs:
                    print(subgraph.node_pairs)
        (tensor([0, 1]), tensor([1, 2]))
        (tensor([2, 3, 4, 5, 4]), tensor([0, 1, 2, 3, 4]))
        (tensor([2, 3, 4]), tensor([0, 1, 2])
        """
        super().__init__(datapipe, fn)
