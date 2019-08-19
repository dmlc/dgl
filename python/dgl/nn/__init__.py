"""Package for neural network common components."""
from __future__ import absolute_import

import sys
import os
import importlib

from . import backend

def _gen_missing_nn_module(module, mod_name):
    def _missing_nn_module(*args, **kwargs):
        raise ImportError('nn.Module "%s" is not supported by backend "%s".'
                          ' You can switch to other backends by setting'
                          ' the DGLBACKEND environment.' % (module, mod_name))
    return _missing_nn_module

def load_backend(mod_name):
    """load backend module according to mod_name
    Parameters
    ----------
    mod_name : str
               The DGL Backend name.
    """
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for nn_module in backend.__dict__:
        if nn_module.startswith('__'):
            # ignore python builtin attributes
            continue
        else:
            # load functions and classes
            if nn_module in mod.__dict__:
                setattr(thismod, nn_module, mod.__dict__[nn_module])
            else:
                setattr(thismod, nn_module, _gen_missing_nn_module(nn_module, mod_name))

load_backend(os.environ.get('DGLBACKEND', 'pytorch').lower())
