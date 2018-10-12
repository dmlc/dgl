import os
__backend__ = os.environ.get('DGLBACKEND', 'mxnet').lower()
if __backend__ == 'numpy':
    from dgl.backend.numpy import *
elif __backend__ == 'pytorch':
    from dgl.backend.pytorch import *
elif __backend__ == 'mxnet':
    from dgl.backend.mxnet import *
else:
    raise Exception("Unsupported backend %s" % __backend__)
