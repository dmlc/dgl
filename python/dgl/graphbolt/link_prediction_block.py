"""Unified data structure for input and ouput of all the stages in loading
process, especially for edge level task."""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from .data_block import DataBlock


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
