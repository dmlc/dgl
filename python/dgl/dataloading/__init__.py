"""Data loading modules"""
from .dataloader import *

from .. import backend as F

if F.get_preferred_backend() == 'pytorch':
    from .pytorch import *
