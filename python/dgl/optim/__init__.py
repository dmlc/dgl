"""dgl optims."""
import importlib
import sys
import os

from ..backend import backend_name
from ..utils import expand_as_pair

def _load_backend(mod_name):
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)

_load_backend(backend_name)
