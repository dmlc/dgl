"""Package for neural network common components."""
import importlib
import sys
import os
from ..backend import backend_name

def _load_backend(mod_name):
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


LOAD_ALL = os.getenv("DGL_LOADALL", "False")

if LOAD_ALL.lower() != "false":
    from .mxnet import *
    from .pytorch import *
    from .tensorflow import *
else:
    _load_backend(backend_name)
