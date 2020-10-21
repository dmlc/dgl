""" API for transferring data to/from the GPU over second stream.A """


from .. import backend as F
from .._ffi.function import _init_api

class Transfer(object):
    """ Class for representing an asynchronous transfer. """
    def __init__(self, transfer_id, handle):
        """ Create a new Transfer object.

        Parameters
        ----------
        transfer_id : int
            The id of the asynchronous tranfer.
        handle : DGLAsyncTransferer
            The handle of the DGLAsyncTransferer object that initiated the
            transfer.
        """

        self._transfer_id = transfer_id
        self._handle = handle

    def wait(self):
        """ Wait for this transfer to finish, and return the result.

        Returns
        -------
        Tensor
            The new tensor on the target context.
        """
        res_tensor = _CAPI_DGLAsyncSamplerWait(self._handle, self._transfer_id)
        return F.zerocopy(F.from_dgl_ndarray(res_tensor))



class AsyncTransferer(object):
    """ Class for initiating asynchronous copies to/from the GPU on a second
    GPU stream. """
    def __init__(self, ctx):
        """ Create a new AsyncTransferer object.

        Parameters
        ----------
        ctx : Device context object.
            The context in which the second stream will be created. Must be a
            GPU context.
        """
        self._handle = _CAPI_DGLAsyncSamplerCreate(ctx)

    def async_copy(self, tensor, ctx):
        """ Initiate an asynchronous copy on the internal stream.

        Parameters
        ----------
        tensor : Tensor
            The tensor to transfer.
        ctx : Device context object.
            The context to transfer to.

        Returns
        -------
        Transfer
            A Transfer object that can be waited on to get the tensor in the
            new context.
        """
        transfer_id = _CAPI_DGLAsyncSamplerStartTransfer(self._handle, tensor, ctx)
        return Transfer(transfer_id, self._handle)


_init_api("dgl.ndarray")
