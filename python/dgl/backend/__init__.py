from __future__ import absolute_import

import sys
import os
import json
import importlib

from . import backend
from .set_default_backend import set_default_backend
from itertools import product

_enabled_apis = set()


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError('API "%s" is not supported by backend "%s".'
                          ' You can switch to other backends by setting'
                          ' the DGLBACKEND environment.' % (api, mod_name))
    return _missing_api

_notes_docstring = r"""
    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics."""

def _gen_sddmm_func(lhs_target, rhs_target, binary_op):
    name = "{}_{}_{}".format(lhs_target, binary_op, rhs_target)
    target_dict = {
        'u': "source node",
        'e': "edge",
        'v': "destination node"
    }
    lhs_str = target_dict[lhs_target]
    rhs_str = target_dict[rhs_target]
    docstring = r"""Generalized SDDMM function.
    It computes edge features by {} {} features and {} features.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {} features.
    y : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.
    {}""".format(binary_op, lhs_str, rhs_str,
                 lhs_str, rhs_str,
                 _notes_docstring)

    def func(g, x, y):
        return gsddmm(g, binary_op, x, y,
                      lhs_target=lhs_target, rhs_target=rhs_target)
    func.__name__ = name
    func.__doc__ = docstring
    return func

def _gen_spmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = """Generalized SpMM function.
    It fuses two steps into one kernel.

    1. Computes messages by {} source node and edge features.
    2. Aggregate the messages by {} as the features on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The source node features.
    y : tensor
        The edge features.

    Returns
    -------
    tensor
        The result tensor.
    {}""".format(binary_op, reduce_op,
                 _notes_docstring)

    def func(g, x, y):
        return gspmm(g, binary_op, reduce_op, x, y)
    func.__name__ = name
    func.__doc__ = docstring
    return func

def _gen_copy_reduce_func(binary_op, reduce_op):

    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies node feature to edge as the message.",
        'copy_e': "It regards edge feature as message."
    }
    x_str = {
        "copy_u": "source node",
        "copy_e": "edge"
    }
    docstring = lambda binary_op: """Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """.format(
        binary_str[binary_op],
        reduce_op,
        x_str[binary_op],
        _notes_docstring)

    def func(g, x):
        if binary_op == 'copy_u':
            return gspmm(g, 'copy_lhs', reduce_op, x, None)
        else:
            return gspmm(g, 'copy_rhs', reduce_op, None, x)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func

def _register_sddmm_func(mod, enabled_apis):
    """Register sddmm functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_sddmm_func(lhs, rhs, binary_op)
                setattr(mod, func.__name__, func)
                enabled_apis.add(func.__name__)

def _register_spmm_func(mod, enabled_apis):
    """Register spmm functions"""
    for binary_op in ["add", "sub", "mul", "div", "copy_u", "copy_e"]:
        for reduce_op in ["sum", "max", "min"]:
            if binary_op.startswith("copy"):
                func = _gen_copy_reduce_func(binary_op, reduce_op)
            else:
                func = _gen_spmm_func(binary_op, reduce_op)
            setattr(mod, func.__name__, func)
            enabled_apis.add(func.__name__)

def copy_u(g, x):
    r"""Generalized SDDMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, 'copy_lhs', x, None)

def copy_v(g, x):
    r"""Generalized SDDMM function that copies destination node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The destination node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, 'copy_rhs', None, x)

def load_backend(mod_name):
    print('Using backend: %s' % mod_name, file=sys.stderr)
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
            # log backend name
            setattr(thismod, 'backend_name', mod_name)
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))
    _register_sddmm_func(thismod, _enabled_apis)
    _register_spmm_func(thismod, _enabled_apis)
    setattr(thismod, copy_u.__name__, copy_u)
    _enabled_apis.add(copy_u.__name__)
    setattr(thismod, copy_v.__name__, copy_v)
    _enabled_apis.add(copy_v.__name__)


def get_preferred_backend():
    config_path = os.path.join(os.path.expanduser('~'), '.dgl', 'config.json')
    backend_name = None
    if "DGLBACKEND" in os.environ:
        backend_name = os.getenv('DGLBACKEND')
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get('backend', '').lower()

    if (backend_name in ['tensorflow', 'mxnet', 'pytorch']):
        return backend_name
    else:
        print("DGL backend not selected or invalid.  "
              "Assuming PyTorch for now.", file=sys.stderr)
        set_default_backend('pytorch')
        return 'pytorch'


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
