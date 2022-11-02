"""DGL type hints"""

__all__ = ["NData", "EData", "NType", "EType"]

from typing import Tuple, Union

try:
    import torch

    NData = torch.Tensor
    EData = torch.Tensor
except ImportError:
    # When PyTorch is not available, create two class stubs to represent
    # EData and NData. TODO(minjie): remove this when DGL becomes PyTorch native.
    class NData:
        """Type hint class to represent node data."""
        pass

    class EData:
        """Type hint class to represent edge data."""
        pass


NType = str
EType = Union[str, Tuple[str, str, str]]
