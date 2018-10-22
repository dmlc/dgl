from __future__ import absolute_import

import os

__backend__ = os.environ.get('DGLBACKEND', 'pytorch').lower()
if __backend__ == 'numpy':
    from .numpy import *
    create_immutable_graph_index=None
elif __backend__ == 'pytorch':
    from .pytorch import *
    create_immutable_graph_index=None
elif __backend__ == 'mxnet':
    from .mxnet import *
    from .mxnet_immutable_graph_index import create_immutable_graph_index
else:
    raise Exception("Unsupported backend %s" % __backend__)
