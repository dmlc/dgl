"""Unified data structure for input and ouput of all the stages in loading process, especially for node level task."""

from dataclasses import dataclass
from typing import Dict, Union

import torch

from ..unified_data_struct import UnifiedDataStruct


@dataclass
class NodeUnifiedDataStruct(UnifiedDataStruct):
    seed_node: Union[torch.Tensor, Dict[str, torch.Tensor]]
    label: Union[torch.Tensor, Dict[str, torch.Tensor]]
