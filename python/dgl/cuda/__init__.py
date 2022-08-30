""" CUDA wrappers """
from .. import backend as F
from . import nccl
if F.get_preferred_backend() == 'pytorch':
    from .streams import *
