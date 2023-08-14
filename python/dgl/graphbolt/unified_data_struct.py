"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from .sampled_subgraph import SampledSubgraph

__all__ = ["UnifiedDataStruct"]


@dataclass
class UnifiedDataStruct:
    r"""A composite data class for data structure in the graphbolt. It is
    designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    sampled_subgraph: SampledSubgraph
    """
    An instance of the 'SampledSubgraph' class, representing a subset of a
    larger graph structure.
    """
    node_feature: Union[torch.Tensor, Dict[str, torch.Tensor]]
    """A representation of node feature.
    - If `node_feature` is a tensor: It indicates the graph is homogeneous.
    - If `node_feature` is a dictionary: The keys should be node type and the
      value should be corresponding node feature.
    """
    edge_feature: Union[torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]]
    """A representation of edge feature.
    - If `edge_feature` is a tensor: It indicates the graph is homogeneous.
    - If `edge_feature` is a dictionary: The keys should be edge type and the
      value should be corresponding edge feature.
    """
