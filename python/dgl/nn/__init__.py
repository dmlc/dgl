"""The ``dgl.nn`` package contains framework-specific implementations for
common Graph Neural Network layers (or module in PyTorch, Block in MXNet).
Users can directly import ``dgl.nn.<layer_name>`` (e.g., ``dgl.nn.GraphConv``),
and the package will dispatch the layer name to the actual implementation
according to the backend framework currently in use.

Note that there are coverage differences among frameworks. If you encounter
an ``ImportError: cannot import name 'XXX'`` error, that means the layer is
not available to the current backend. If you wish a module to appear in DGL,
please `create an issue <https://github.com/dmlc/dgl/issues>`_ started with
"[Feature Request] NN Module XXXModel". If you want to contribute a NN module,
please `create a pull request <https://github.com/dmlc/dgl/pulls>`_ started
with "[NN] XXX module".
"""

import importlib
import sys
import os

# [BarclayII] Not sure what's going on with pylint.
# Possible issue: https://github.com/PyCQA/pylint/issues/2648
from . import functional        # pylint: disable=import-self

from ..backend import backend_name
from ..utils import expand_as_pair

def _load_backend(mod_name):
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)

_load_backend(backend_name)
