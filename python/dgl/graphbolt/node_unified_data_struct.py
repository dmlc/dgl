"""Unified data structure for input and ouput of all the stages in loading
process, especially for node level task."""

from dataclasses import dataclass
from typing import Dict, Union

import torch

from .unified_data_struct import UnifiedDataStruct


@dataclass
class NodeUnifiedDataStruct(UnifiedDataStruct):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling node level
    tasks."""

    label: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seed nodes in the graph.
    - If `label` is a tensor: It indicates the graph is homogeneous.
    - If `label` is a dictionary: The keys should be node type and the
      value should be corresponding node labels to given 'seed_node'.
    """
