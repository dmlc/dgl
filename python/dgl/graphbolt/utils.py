"""Utility functions for external use."""

from typing import Dict, Tuple, Union

import torch

from .minibatch import MiniBatch


def add_reverse_edges(
    edges: Union[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
    ],
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
    edges : Union[Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                Tuple[torch.Tensor, torch.Tensor]]
        - If sampled subgraph is homogeneous, then `edges` should be a pair of
        of tensors.
        - If sampled subgraph is heterogeneous, then `edges` should be a
        dictionary of edge types and the corresponding edges to exclude.
    reverse_etypes_mapping : Dict[str, str], optional
        The mapping from the original edge types to their reverse edge types.

    Returns
    -------
    Union[Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor]]
        The node pairs contain both the original edges and their reverse
        counterparts.

    Examples
    --------
    >>> edges = {"A:r:B": (torch.tensor([0, 1]), torch.tensor([1, 2]))}
    >>> print(gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"}))
    {'A:r:B': (tensor([0, 1]), tensor([1, 2])),
    'B:rr:A': (tensor([1, 2]), tensor([0, 1]))}

    >>> edges = (torch.tensor([0, 1]), torch.tensor([2, 1]))
    >>> print(gb.add_reverse_edges(edges))
    (tensor([0, 1, 2, 1]), tensor([2, 1, 0, 1]))
    """
    if isinstance(edges, tuple):
        u, v = edges
        return (torch.cat([u, v]), torch.cat([v, u]))
    else:
        combined_edges = edges.copy()
        for etype, reverse_etype in reverse_etypes_mapping.items():
            if etype in edges:
                if reverse_etype in combined_edges:
                    u, v = combined_edges[reverse_etype]
                    u = torch.cat([u, edges[etype][1]])
                    v = torch.cat([v, edges[etype][0]])
                    combined_edges[reverse_etype] = (u, v)
                else:
                    combined_edges[reverse_etype] = (
                        edges[etype][1],
                        edges[etype][0],
                    )
        return combined_edges


def add_reverse_edges_2(
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
):
    """
    Exclude seed edges with or without their reverse edges from the sampled
    subgraphs in the minibatch.

    Parameters
    ----------
    minibatch : MiniBatch
        The minibatch.
    reverse_etypes_mapping : Dict[str, str] = None
        The mapping from the original edge types to their reverse edge types.
    """
    if minibatch.node_pairs is not None:
        edges_to_exclude = minibatch.node_pairs
        if include_reverse_edges:
            edges_to_exclude = add_reverse_edges(
                minibatch.node_pairs, reverse_etypes_mapping
            )
    else:
        edges_to_exclude = minibatch.seeds
        if include_reverse_edges:
            edges_to_exclude = add_reverse_edges_2(
                edges_to_exclude, reverse_etypes_mapping
            )
    minibatch.sampled_subgraphs = [
        subgraph.exclude_edges(edges_to_exclude)
        for subgraph in minibatch.sampled_subgraphs
    ]
    return minibatch
