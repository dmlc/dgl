""" API for transferring data to/from the GPU over second stream.A """


from .. import backend as F
from .. import ndarray
from .. import utils
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
        res_tensor = _CAPI_DGLAsyncTransfererWait(self._handle, self._transfer_id)
        return F.zerocopy_from_dgl_ndarray(res_tensor)



class AsyncTransferer(object):
    """ Class for initiating asynchronous copies to/from the GPU on a second
    GPU stream. """
    def __init__(self, device):
        """ Create a new AsyncTransferer object.

        Parameters
        ----------
        device : Device or context object.
            The context in which the second stream will be created. Must be a
            GPU context for the copy to be asynchronous.
        """
        if isinstance(device, ndarray.DGLContext):
            ctx = device
        else:
            ctx = utils.to_dgl_context(device)
        self._handle = _CAPI_DGLAsyncTransfererCreate(ctx)

    def async_copy(self, tensor, device):
        """ Initiate an asynchronous copy on the internal stream.

        Parameters
        ----------
        tensor : Tensor
            The tensor to transfer.
        device : Device or context object.
            The context to transfer to.

        Returns
        -------
        Transfer
            A Transfer object that can be waited on to get the tensor in the
            new context.
        """
        if isinstance(device, ndarray.DGLContext):
            ctx = device
        else:
            ctx = utils.to_dgl_context(device)

        tensor = F.zerocopy_to_dgl_ndarray(tensor)

        transfer_id = _CAPI_DGLAsyncTransfererStartTransfer(self._handle, tensor, ctx)
        return Transfer(transfer_id, self._handle)


_init_api("dgl.dataloading.async_transferer")
