from ._ffi.function import _init_api

def cb1(x):
    print('Inside callback. Got value', x)

def test_callback():
    _CAPI_DGLCallbackTestAPI(cb1)

def test_async_callback():
    _CAPI_DGLAsyncCallbackTestAPI(cb1)

_init_api("dgl.test")
