"""DGL type hints"""

__all__ = ['NData', 'EData', 'NType', 'EType']

from typing import Union, Tuple

try:
    import torch
    NData = torch.Tensor
    EData = torch.Tensor
except:
    # When PyTorch is not available, create two class stubs to represent
    # EData and NData. TODO(minjie): remove this when DGL becomes PyTorch native.
    class NData: pass
    class EData: pass

NType = str
EType = Union[str, Tuple[str, str, str]]
