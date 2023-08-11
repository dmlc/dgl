"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from .sampled_subgraph import SampledSubgraph

__all__ = ["UnifiedDataStruct"]


@dataclass
class UnifiedDataStruct:
    sampled_subgraph: SampledSubgraph
    node_feature: Union[torch.Tensor, Dict[str, torch.Tensor]]
    edge_feature: Union[torch.Tensor, Dict[Tuple(str, str, str), torch.Tensor]]
