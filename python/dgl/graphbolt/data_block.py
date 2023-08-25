"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

from .sampled_subgraph import SampledSubgraph

__all__ = ["DataBlock"]


@dataclass
class DataBlock:
    r"""A composite data class for data structure in the graphbolt. It is
    designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    sampled_subgraphs: List[SampledSubgraph] = None
    """
    A list of 'SampledSubgraph's, each one corresponding to one layer,
    representing a subset of a larger graph structure.
    """

    node_feature: Dict[Tuple[str, str], torch.Tensor] = None
    """
    A representation of node features.
    Keys are tuples of '(node_type, feature_name)' and the values are
    corresponding features. Note that for a homogeneous graph, where there are
    no node types, 'node_type' should be None.
    """

    edge_feature: List[Dict[Tuple[str, str], torch.Tensor]] = None
    """Edge features associated with the 'sampled_subgraphs'.
    The keys are tuples in the format '(edge_type, feature_name)', and the
    values represent the corresponding features. In the case of a homogeneous
    graph where no edge types exist, 'edge_type' should be set to None.
    Note 'edge_type' are of format 'str-str-str'.
    """

    input_nodes: Union[
        torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]
    ] = None
    """A representation of input nodes in the outermost layer. Conatins all nodes
       in the 'sampled_subgraphs'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
    """
