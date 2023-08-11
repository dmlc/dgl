"""Unified data structure for input and ouput of all the stages in loading process, especially for edge level task."""

from dataclasses import dataclass
from ..unified_data_struct import UnifiedDataStruct
import torch
from typing import Dict, Tuple, Union
from ..data_format import LinkPredictionEdgeFormat


@dataclass
class LinkUnifiedDataStruct(UnifiedDataStruct):
  node_pair: Union[Tuple[torch.Tensor, torch.Tensor], Dict[Tuple(str, str, str), Tuple[torch.Tensor, torch.Tensor]]]
  label: Union[torch.Tensor, Dict[str, torch.Tensor]]
  negative_head: Union[torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]]
  negative_tail: Union[torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]]
  data_format: LinkPredictionEdgeFormat
    
    
                                
