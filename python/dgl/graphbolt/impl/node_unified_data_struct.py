"""Unified data structure for input and ouput of all the stages in loading process, especially for node level task."""

from dataclasses import dataclass
from ..unified_data_struct import UnifiedDataStruct
import torch
from typing import Dict, Union



@dataclass
class NodeUnifiedDataStruct(UnifiedDataStruct):
  seed_node: Union[torch.Tensor, Dict[str, torch.Tensor]]
  label: Union[torch.Tensor, Dict[str, torch.Tensor]]
    
    
                                
