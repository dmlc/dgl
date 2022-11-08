from __future__ import absolute_import

import importlib
import json
import logging
import os
import sys

from . import backend
from .set_default_backend import set_default_backend

_enabled_apis = set()

logger = logging.getLogger("dgl-core")


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError(
            'API "%s" is not supported by backend "%s".'
            " You can switch to other backends by setting"
            " the DGLBACKEND environment." % (api, mod_name)
        )

    return _missing_api


def load_backend(mod_name):
    # Load backend does four things:
    # (1) Import backend framework (PyTorch, MXNet, Tensorflow, etc.)
    # (2) Import DGL C library.  DGL imports it *after* PyTorch/MXNet/Tensorflow.  Otherwise
    #     DGL will crash with errors like `munmap_chunk(): invalid pointer`.
    # (3) Sets up the tensoradapter library path.
    # (4) Import the Python wrappers of the backend framework.  DGL does this last because
    #     it already depends on both the backend framework and the DGL C library.
    if mod_name == "pytorch":
        import torch

        mod = torch
    elif mod_name == "mxnet":
        import mxnet

        mod = mxnet
    elif mod_name == "tensorflow":
        import tensorflow

        mod = tensorflow
    else:
        raise NotImplementedError("Unsupported backend: %s" % mod_name)

    from .._ffi.base import load_tensor_adapter  # imports DGL C library

    version = mod.__version__
    load_tensor_adapter(mod_name, version)

    logger.debug("Using backend: %s" % mod_name)
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for'
                    ' backend "%s".' % (mod_name)
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, "data_type_dict", data_type_dict)

            # for data types with aliases, treat the first listed type as
            # the true one
            rev_data_type_dict = {}
            for k, v in data_type_dict.items():
                if not v in rev_data_type_dict.keys():
                    rev_data_type_dict[v] = k
            setattr(thismod, "reverse_data_type_dict", rev_data_type_dict)
            # log backend name
            setattr(thismod, "backend_name", mod_name)
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))


def get_preferred_backend():
    default_dir = None
    if "DGLDEFAULTDIR" in os.environ:
        default_dir = os.getenv("DGLDEFAULTDIR")
    else:
        default_dir = os.path.join(os.path.expanduser("~"), ".dgl")
    config_path = os.path.join(default_dir, "config.json")
    backend_name = None
    if "DGLBACKEND" in os.environ:
        backend_name = os.getenv("DGLBACKEND")
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get("backend", "").lower()

    if backend_name in ["tensorflow", "mxnet", "pytorch"]:
        return backend_name
    else:
        print(
            "DGL backend not selected or invalid.  "
            "Assuming PyTorch for now.",
            file=sys.stderr,
        )
        set_default_backend(default_dir, "pytorch")
        return "pytorch"


load_backend(get_preferred_backend())


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


def to_dgl_nd(data):
    return zerocopy_to_dgl_ndarray(data)


def from_dgl_nd(data):
    return zerocopy_from_dgl_ndarray(data)
