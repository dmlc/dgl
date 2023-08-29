"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

from .sampled_subgraph import SampledSubgraph

__all__ = ["DataBlock", "NodeClassificationBlock", "LinkPredictionBlock"]


@dataclass
class DataBlock:
    r"""A composite data class for data structure in the graphbolt. It is
    designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    sampled_subgraphs: List[SampledSubgraph] = None
    """A list of 'SampledSubgraph's, each one corresponding to one layer,
    representing a subset of a larger graph structure.
    """

    node_feature: Dict[Tuple[str, str], torch.Tensor] = None
    """A representation of node features.
    Keys are tuples of '(node_type, feature_name)' and the values are
    corresponding features. Note that for a homogeneous graph, where there are
    no node types, 'node_type' should be None.
    """

    edge_feature: List[Dict[Tuple[str, str], torch.Tensor]] = None
    """Edge features associated with the 'sampled_subgraphs'.
    The keys are tuples in the format '(edge_type, feature_name)', and the
    values represent the corresponding features. In the case of a homogeneous
    graph where no edge types exist, 'edge_type' should be set to None.
    """

    input_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """A representation of input nodes in the outermost layer. Conatins all nodes
       in the 'sampled_subgraphs'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
    """


@dataclass
class NodeClassificationBlock(DataBlock):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling node level
    tasks."""

    seed_node: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of seed nodes used for sampling in the graph.
    - If `seed_node` is a tensor: It indicates the graph is homogeneous.
    - If `seed_node` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node ids.
    """

    label: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seed nodes in the graph.
    - If `label` is a tensor: It indicates the graph is homogeneous.
    - If `label` is a dictionary: The keys should be node type and the
      value should be corresponding node labels to given 'seed_node'.
    """


@dataclass
class LinkPredictionBlock(DataBlock):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling edge level
    tasks."""

    node_pair: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of seed node pairs utilized in link prediction tasks.
    - If `node_pair` is a tuple: It indicates a homogeneous graph where each
      tuple contains two tensors representing source-destination node pairs.
    - If `node_pair` is a dictionary: The keys should be edge type, and the
      value should be a tuple of tensors representing node pairs of the given
      type.
    """

    label: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with the link prediction task.
    - If `label` is a tensor: It indicates a homogeneous graph. The value are
      edge labels corresponding to given 'node_pair'.
    - If `label` is a dictionary: The keys should be edge type, and the value
      should correspond to given 'node_pair'.
    """

    negative_head: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the head nodes in the link
    prediction task.
    - If `negative_head` is a tensor: It indicates a homogeneous graph.
    - If `negative_head` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    negative_tail: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the tail nodes in the link
    prediction task.
    - If `negative_tail` is a tensor: It indicates a homogeneous graph.
    - If `negative_tail` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    compacted_node_pair: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of compacted node pairs corresponding to 'node_pair', where
    all node ids inside are compacted.
    """

    compacted_negative_head: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_head', where
    all node ids inside are compacted.
    """

    compacted_negative_tail: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_tail', where
    all node ids inside are compacted.
    """
