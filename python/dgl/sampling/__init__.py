"""Sampling operators.

This module contains the implementations of various sampling operators.
"""
from .randomwalks import *
from .pinsage import *
from .neighbor import *
from .dataloader import *

from .. import backend as F

if F.get_preferred_backend() == 'pytorch':
    from .pytorch import *
