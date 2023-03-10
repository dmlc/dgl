""" CUDA wrappers """
from .. import backend as F

if F.get_preferred_backend() == "pytorch":
    from . import nccl
