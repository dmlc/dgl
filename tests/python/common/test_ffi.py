import os
import unittest

import backend as F

import dgl
import numpy as np
import pytest


@unittest.skipIf(os.name == "nt", reason="Cython only works on linux")
def test_cython():
    import dgl._ffi._cy3.core


@pytest.mark.parametrize("arg", [1, 2.3])
def test_callback(arg):
    def cb(x):
        return x + 1

    ret = dgl._api_internal._TestPythonCallback(cb, arg)
    assert ret == arg + 1


@pytest.mark.parametrize("dtype", [F.float32, F.float64, F.int32, F.int64])
def _test_callback_array(dtype):
    def cb(x):
        return F.to_dgl_nd(F.from_dgl_nd(x) + 1)

    arg = F.copy_to(F.tensor([1, 2, 3], dtype=dtype), F.ctx())
    ret = F.from_dgl_nd(
        dgl._api_internal._TestPythonCallback(cb, F.to_dgl_nd(arg))
    )
    assert np.allclose(F.asnumpy(ret), F.asnumpy(arg) + 1)


@pytest.mark.parametrize("arg", [1, 2.3])
def test_callback_thread(arg):
    def cb(x):
        return x + 1

    ret = dgl._api_internal._TestPythonCallbackThread(cb, arg)
    assert ret == arg + 1


@pytest.mark.parametrize("dtype", [F.float32, F.float64, F.int32, F.int64])
def _test_callback_array_thread(dtype):
    def cb(x):
        return F.to_dgl_nd(F.from_dgl_nd(x) + 1)

    arg = F.copy_to(F.tensor([1, 2, 3], dtype=dtype), F.ctx())
    ret = F.from_dgl_nd(
        dgl._api_internal._TestPythonCallbackThread(cb, F.to_dgl_nd(arg))
    )
    assert np.allclose(F.asnumpy(ret), F.asnumpy(arg) + 1)
