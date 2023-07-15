"""Feature storage classes for DataLoading"""
from .. import backend as F
from .base import *
from .numpy import *

# Defines the name TensorStorage
if F.get_preferred_backend() == "pytorch":
    from .pytorch_tensor import PyTorchTensorStorage as TensorStorage
else:
    from .tensor import BaseTensorStorage as TensorStorage
