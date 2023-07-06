"""Negative samplers"""


from collections.abc import Mapping

import torch

from .graph_storage import CSCSamplingGraph
from .linked_data_format import LinkedDataFormat

__all__ = ["PerSourceUniformSampler"]


class _BaseNegativeSampler:
    """
    A negative sampler used to generate negative samples and return
    a mix of positive and negative samples, the format of the output
    depends on the specified `linked_data_format`.
    """

    def __init__(
        self,
        graph: CSCSamplingGraph,
        negative_ratio: int,
        linked_data_format: LinkedDataFormat,
    ):
        """
        Initlization for a negative sampler.

        Parameters
        ----------
        negative_ratio : int
            The proportion of negative samples to positive samples.
        linked_data_format : LinkedDataFormat
            Determines the format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges,  while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).
        """
        super().__init__()
        self.graph = graph
        assert (
            negative_ratio > 0
        ), "Negative_ratio should shoubld be positive Integer."
        self.negative_ratio = negative_ratio
        assert linked_data_format in [
            LinkedDataFormat.CONDITIONED,
            LinkedDataFormat.INDEPENDENT,
        ], f"Unsupported data format: {linked_data_format}."
        self.linked_data_format = linked_data_format

    def _generate(self, pos_edges, etype=None):
        raise NotImplementedError

    def __call__(self, pos_edges):
        """
        Generates a mix of positive and negative samples, the format of which
        depends on the specified `linked_data_format`.

        Parameters
        ----------
        pos_edges : List[Tensor] or Dict[etype, List[Tensor]]
            Represents source-destination node pairs of positive edges, where
            positive means the edge must exist in the graph.

        Returns
        -------
        List[Tensor] or Dict[etype, List[Tensor]]
            A collection of edges or a dictionary that maps etypes to lists of
            edges which includes both positive and negative samples. The format
            of it is determined by the provided 'linked_data_format'.
        """

        def generate_negative_pairs(pos_edge, etype):
            neg_src, neg_dst = self._generate(pos_edge, etype)
            pos_src, pos_dst = pos_edge
            if self.linked_data_format == LinkedDataFormat.INDEPENDENT:
                pos_label = torch.ones_like(pos_src)
                neg_label = torch.zeros_like(neg_src)
                src = torch.cat([pos_src, neg_src])
                dst = torch.cat([pos_dst, neg_dst])
                labels = torch.cat([pos_label, neg_label])
                return (src, dst, labels)
            else:
                neg_src = neg_src.view(-1, self.negative_ratio)
                neg_dst = neg_dst.view(-1, self.negative_ratio)
                return (pos_src, pos_dst, neg_src, neg_dst)

        if isinstance(pos_edges, Mapping):
            return {
                etype: generate_negative_pairs(pos_edge, etype)
                for etype, pos_edge in pos_edges.items()
            }
        else:
            return generate_negative_pairs(pos_edges, None)


class PerSourceUniformSampler(_BaseNegativeSampler):
    """Negative samplers randomly select negative destination nodes for each
    source node based on a uniform distribution. It's important to note that
    the term 'negative' refers to false negatives, indicating that the sampled
    pairs are not ensured to be absent in the graph.

    For each edge ``(u, v)``, it is supposed to generate `negative_ratio` pairs
    of negative edges ``(u, v')``, where ``v'`` is chosen uniformly from all
    the nodes in the graph.

    Examples
    --------
    >>> from dgl import graphbolt as gb
    >>> indptr = torch.LongTensor([0, 2, 4, 5])
    >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
    >>> graph = gb.from_csc(indptr, indices)
    >>> pos_edges = (torch.tensor([0, 1]), torch.tensor([1, 2]))
    >>> linked_data_format = gb.LinkedDataFormat.INDEPENDENT
    >>> neg_sampler = gb.PerSourceUniformSampler(graph, 1, linked_data_format)
    >>> neg_sampler(pos_edges)
    (tensor([0, 1, 0, 1]), tensor([1, 2, 1, 0]), tensor([1, 1, 0, 0]))
    """

    def _generate(self, pos_edges, etype=None):
        return self.graph.sample_negative_edges_uniform(
            etype,
            pos_edges,
            self.negative_ratio,
        )
