from __future__ import absolute_import

import os

__backend__ = os.environ.get('DGLBACKEND', 'pytorch').lower()
if __backend__ == 'numpy':
    from .numpy import *
elif __backend__ == 'pytorch':
    from .pytorch import *
else:
    raise Exception("Unsupported backend %s" % __backend__)
