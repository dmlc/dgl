"""Package for neural network common components."""
import os
import importlib
import sys

def load_backend(mod_name):
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)

load_backend(os.environ.get('DGLBACKEND', 'pytorch').lower())
