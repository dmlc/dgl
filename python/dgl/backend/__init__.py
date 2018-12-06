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

def load_backend(mod_name):
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api in backend.__dict__.keys():
        if api.startswith('__'):
            # ignore python builtin attributes
            continue
        if api == 'data_type_dict':
            # load data type
            if api not in mod.__dict__:
                raise ImportError('API "data_type_dict" is required but missing for'
                                  ' backend "%s".' % (mod_name))
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, 'data_type_dict', data_type_dict)
            setattr(thismod,
                    'reverse_data_type_dict',
                    {v: k for k, v in data_type_dict.items()})
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))

load_backend(os.environ.get('DGLBACKEND', 'pytorch').lower())

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
