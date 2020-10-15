
from ._ffi.function import _init_api

class Transfer(object):
    def __init__(self, transfer_id, handle):
        self._transfer_id = transfer_id
        self._handle = handle

    def wait(self):
        res_tensor = _CAPI_DGLAsyncSamplerWait(self._handle,
                self._transfer_id)
        return F.zerocopy(from_dgl_ndarray(res_tensor)



class AsyncTransferer(object):
    def __init__(self, ctx):
        self._handle = _CAPI_DGLAsyncSamplerCreate(ctx)

    def async_copy(self, tensor, ctx):
        transfer_id = _CAPI_DGLAsyncSamplerStartTransfer(self._handle, tensor, ctx)
        return Transfer(transfer_id, self._handle)


_init_api("dgl.ndarray")
