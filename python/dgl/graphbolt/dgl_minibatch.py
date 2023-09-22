"""DGL minibatch."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

from dgl.heterograph import DGLBlock


__all__ = ["DGLMiniBatch"]


@dataclass
class DGLMiniBatch:
    r"""A data class designed for the DGL library, encompassing all the
    necessary fields for computation using the DGL library.."""

    blocks: List[DGLBlock] = None
    """A list of 'DGLBlock's, each one corresponding to one layer, representing
    a bipartite graph used for message passing.
    """

    input_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """A representation of input nodes in the outermost layer. Conatins all nodes
       in the 'sampled_subgraphs'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
    """

    output_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of output nodes, usually also the seed nodes, used for
    sampling in the graph.
    - If `output_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `output_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node ids.
    """

    node_features: Union[
        Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]
    ] = None
    """A representation of node features.
      - If keys are single strings: It means the graph is homogeneous, and the
      keys are feature names.
      - If keys are tuples: It means the graph is heterogeneous, and the keys
      are tuples of '(node_type, feature_name)'.
    """

    edge_features: List[
        Union[Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]]
    ] = None
    """Edge features associated with the 'sampled_subgraphs'.
      - If keys are single strings: It means the graph is homogeneous, and the
      keys are feature names.
      - If keys are tuples: It means the graph is heterogeneous, and the keys
      are tuples of '(edge_type, feature_name)'. Note, edge type is a triplet
      of format (str, str, str).
    """

    labels: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seed nodes / node pairs in the graph.
    - If `labels` is a tensor: It indicates the graph is homogeneous. The value
      are corresponding labels to given 'output_nodes' or 'node_pairs'.
    - If `labels` is a dictionary: The keys are node or edge type and the value
      should be corresponding labels to given 'output_nodes' or 'node_pairs'.
    """

    positive_node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of positive graphs used for evaluating or computing loss in
    link prediction tasks.
    - If `positive_node_pairs` is a tuple: It indicates a homogeneous graph
    containing two tensors representing source-destination node pairs.
    - If `positive_node_pairs` is a dictionary: The keys should be edge type,
    and the value should be a tuple of tensors representing node pairs of the
    given type.
    """

    negative_node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of negative graphs used for evaluating or computing loss in
    link prediction tasks.
    - If `negative_node_pairs` is a tuple: It indicates a homogeneous graph
    containing two tensors representing source-destination node pairs.
    - If `negative_node_pairs` is a dictionary: The keys should be edge type,
    and the value should be a tuple of tensors representing node pairs of the
    given type.
    """
