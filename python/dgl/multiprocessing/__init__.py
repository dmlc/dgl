"""Wrapper of the multiprocessing module for multi-GPU training."""

# To avoid duplicating the graph structure for node classification or link prediction
# training we recommend using fork() rather than spawn() for multiple GPU training.
# However, we need to work around https://github.com/pytorch/pytorch/issues/17199 to
# make fork() and openmp work together.
from .. import backend as F

if F.get_preferred_backend() == "pytorch":
    # Wrap around torch.multiprocessing...
    from torch.multiprocessing import *

    # ... and override the Process initializer.
    from .pytorch import *
else:
    # Just import multiprocessing module.
    from multiprocessing import *  # pylint: disable=redefined-builtin
