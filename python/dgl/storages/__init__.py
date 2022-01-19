from .. import backend as F

if F.get_preferred_backend() == 'pytorch':
    from .pytorch_tensor import *
else:
    from .tensor import *
from .base import *
from .numpy import *
