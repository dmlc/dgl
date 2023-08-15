"""Unified data structure for input and ouput of all the stages in loading
process, especially for edge level task."""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from ..data_format import LinkPredictionEdgeFormat
from ..unified_data_struct import UnifiedDataStruct


@dataclass
class LinkUnifiedDataStruct(UnifiedDataStruct):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling edge level
    tasks."""

    node_pair: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[Tuple(str, str, str), Tuple[torch.Tensor, torch.Tensor]],
    ]
    """
    Representation of node pairs in the graph that require link prediction.
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

    negative_head: Union[
        torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]
    ] = None
    """
    Representation of negative samples for the head nodes in the link
    prediction task.
    - If `negative_head` is a tensor: It indicates a homogeneous graph.
    - If `negative_head` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    For different data formats:
    - If 'data_format' is CONDITIONED, it collaborates with corresponding
      tail nodes in 'node_pair' to creare negative edges.
    - If 'data_format' is HEAD_CONDITIONED, it collaborates with corresponding
      tail nodes in 'negative_tail' to creare negative edges. And both are of
      same shape.
    - otherwise, this field should be empty.
    """

    negative_tail: Union[
        torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]
    ] = None
    """
    Representation of negative samples for the tail nodes in the link
    prediction task.
    - If `negative_tail` is a tensor: It indicates a homogeneous graph.
    - If `negative_tail` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    For different data formats:
    - If 'data_format' is CONDITIONED, it collaborates with corresponding
      head nodes in 'node_pair' to creare negative edges.
    - If 'data_format' is TAIL_CONDITIONED, it collaborates with corresponding
      head nodes in 'negative_head' to creare negative edges. And both are of
      same shape.
    - otherwise, this field should be empty.
    """

    data_format: LinkPredictionEdgeFormat = None
    """
    An instance of the LinkPredictionEdgeFormat class, representing the format
    of edge data used in the link prediction task.
    - If 'data_format' is None, it indicates there are no negative edges. Both
      'negative_head' and 'negative_tail' should be empty.
    - If 'data_format' is CONDITIONED, Both 'negative_head' and 'negative_tail'
      should be non-empty.
    - If 'data_format' is HEAD-CONDITIONED, 'negative_head' should be non-empty
      while 'negative_tail' be empty.
    - If 'data_format' is TAIL-CONDITIONED, 'negative_tail' should be non-empty
      while 'negative_head' be empty.
    """
