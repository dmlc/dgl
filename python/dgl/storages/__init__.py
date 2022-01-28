"""Feature storage classes for DataLoading"""
from .. import backend as F

from .base import *
from .numpy import *
if F.get_preferred_backend() == 'pytorch':
    from .pytorch_tensor import *
else:
    from .tensor import *
