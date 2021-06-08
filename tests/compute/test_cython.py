import dgl
import unittest
import os


@unittest.skipIf(os.name == 'nt', reason='Cython only works on linux')
def test_cython():
    import dgl._ffi._cy3.core
