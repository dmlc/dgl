import os
# __backend__ = 'numpy'
__backend__ = os.environ.get('DGLBACKEND', 'pytorch').lower()
if __backend__ == 'numpy':
    from dgl.backend.numpy import *
elif __backend__ == 'pytorch':
    from dgl.backend.pytorch import *
else:
    raise Exception("Unsupported backend %s" % __backend__)
