"""Utility functions for external use."""
from functools import partial
from typing import Dict, Union

import torch

from torch.utils.data import functional_datapipe

from .minibatch import MiniBatch
from .minibatch_transformer import MiniBatchTransformer


@functional_datapipe("exclude_seed_edges")
class SeedEdgesExcluder(MiniBatchTransformer):
    """A mini-batch transformer used to manipulate mini-batch.

    Functional name: :obj:`transform`.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    include_reverse_edges : bool
        Whether reverse edges should be excluded as well. Default is False.
    reverse_etypes_mapping : Dict[str, str] = None
        The mapping from the original edge types to their reverse edge types.
    asynchronous: bool
        Boolean indicating whether edge exclusion stages should run on
        background threads to hide the latency of CPU GPU synchronization.
        Should be enabled only when sampling on the GPU.
    """

    def __init__(
        self,
        datapipe,
        include_reverse_edges: bool = False,
        reverse_etypes_mapping: Dict[str, str] = None,
        asynchronous=False,
    ):
        exclude_seed_edges_fn = partial(
            exclude_seed_edges,
            include_reverse_edges=include_reverse_edges,
            reverse_etypes_mapping=reverse_etypes_mapping,
            async_op=asynchronous,
        )
        datapipe = datapipe.transform(exclude_seed_edges_fn)
        if asynchronous:
            datapipe = datapipe.buffer()
            datapipe = datapipe.transform(self._wait_for_sampled_subgraphs)
        super().__init__(datapipe)

    @staticmethod
    def _wait_for_sampled_subgraphs(minibatch):
        minibatch.sampled_subgraphs = [
            subgraph.wait() for subgraph in minibatch.sampled_subgraphs
        ]
        return minibatch


def add_reverse_edges(
    edges: Union[Dict[str, torch.Tensor], torch.Tensor],
    reverse_etypes_mapping: Dict[str, str] = None,
):
    r"""
    This function finds the reverse edges of the given `edges` and returns the
    composition of them. In a homogeneous graph, reverse edges have inverted
    source and destination node IDs. While in a heterogeneous graph, reversing
    also involves swapping node IDs and their types. This function could be
    used before `exclude_edges` function to help find targeting edges.
    Note: The found reverse edges may not really exists in the original graph.
    And repeat edges could be added becasue reverse edges may already exists in
    the `edges`.

    Parameters
    ----------
    edges : Union[Dict[str, torch.Tensor], torch.Tensor]
      - If sampled subgraph is homogeneous, then `edges` should be a N*2
        tensors.
      - If sampled subgraph is heterogeneous, then `edges` should be a
        dictionary of edge types and the corresponding edges to exclude.
    reverse_etypes_mapping : Dict[str, str], optional
        The mapping from the original edge types to their reverse edge types.

    Returns
    -------
    Union[Dict[str, torch.Tensor], torch.Tensor]
        The node pairs contain both the original edges and their reverse
        counterparts.

    Examples
    --------
    >>> edges = {"A:r:B": torch.tensor([[0, 1],[1, 2]]))}
    >>> print(gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"}))
    {'A:r:B': torch.tensor([[0, 1],[1, 2]]),
    'B:rr:A': torch.tensor([[1, 0],[2, 1]])}

    >>> edges = torch.tensor([[0, 1],[1, 2]])
    >>> print(gb.add_reverse_edges(edges))
    torch.tensor([[1, 0],[2, 1]])
    """
    if isinstance(edges, torch.Tensor):
        assert edges.ndim == 2 and edges.shape[1] == 2, (
            "Only tensor with shape N*2 is supported now, but got "
            + f"{edges.shape}."
        )
        reverse_edges = edges.flip(dims=(1,))
        return torch.cat((edges, reverse_edges))
    else:
        combined_edges = edges.copy()
        for etype, reverse_etype in reverse_etypes_mapping.items():
            if etype in edges:
                assert edges[etype].ndim == 2 and edges[etype].shape[1] == 2, (
                    "Only tensor with shape N*2 is supported now, but got "
                    + f"{edges[etype].shape}."
                )
                if reverse_etype in combined_edges:
                    combined_edges[reverse_etype] = torch.cat(
                        (
                            combined_edges[reverse_etype],
                            edges[etype].flip(dims=(1,)),
                        )
                    )
                else:
                    combined_edges[reverse_etype] = edges[etype].flip(dims=(1,))
        return combined_edges


def exclude_seed_edges(
    minibatch: MiniBatch,
    include_reverse_edges: bool = False,
    reverse_etypes_mapping: Dict[str, str] = None,
    async_op: bool = False,
):
    """
    Exclude seed edges with or without their reverse edges from the sampled
    subgraphs in the minibatch.

    Parameters
    ----------
    minibatch : MiniBatch
        The minibatch.
    include_reverse_edges : bool
        Whether reverse edges should be excluded as well. Default is False.
    reverse_etypes_mapping : Dict[str, str] = None
        The mapping from the original edge types to their reverse edge types.
    async_op: bool
        Boolean indicating whether the call is asynchronous. If so, the result
        can be obtained by calling wait on the modified sampled_subgraphs.
    """
    edges_to_exclude = minibatch.seeds
    if include_reverse_edges:
        edges_to_exclude = add_reverse_edges(
            edges_to_exclude, reverse_etypes_mapping
        )
    minibatch.sampled_subgraphs = [
        subgraph.exclude_edges(edges_to_exclude, async_op=async_op)
        for subgraph in minibatch.sampled_subgraphs
    ]
    return minibatch
