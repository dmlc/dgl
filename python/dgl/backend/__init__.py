from __future__ import absolute_import

import sys, os
import importlib

from . import backend

_enabled_apis = set()

def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError('API "%s" is not supported by backend "%s".'
                          ' You can switch to other backends by setting'
                          ' the DGLBACKEND environment.' % (api, mod_name))
    return _missing_api

def _load_backend():
    mod_name = os.environ.get('DGLBACKEND', 'pytorch').lower()
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api in backend.__dict__.keys():
        if api == 'data_type_dict':
            # load data type
            if api not in mod.__dict__:
                raise ImportError('API "data_type_dict" is required but missing for'
                                  ' backend "%s".' % (mod_name))
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))

_load_backend()

def is_enabled(api):
    """Return true if the api is enabled by the current backend.

    Parameters
    ----------
    api : str
        The api name.

    Returns
    -------
    bool
        True if the API is enabled by the current backend.
    """
    return api in _enabled_apis
