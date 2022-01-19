"""API for transferring data to the GPU over second stream."""

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
            The id of the asynchronous transfer.
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
    """ Class for initiating asynchronous copies to the GPU on a second
    GPU stream.

    To initiate a transfer to a GPU:

    >>> tensor_cpu = torch.ones(100000).pin_memory()
    >>> transferer = dgl.dataloading.AsyncTransferer(torch.device(0))
    >>> future = transferer.async_copy(tensor_cpu, torch.device(0))

    And then to wait for the transfer to finish and get a copy of the tensor on
    the GPU.

    >>> tensor_gpu = future.wait()


    """
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
        """ Initiate an asynchronous copy on the internal stream. For this call
        to be asynchronous, the context the AsyncTranserer is created with must
        be a GPU context, and the input tensor must be in pinned memory.

        Currently, only transfers to the GPU are supported.

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

        if ctx.device_type != ndarray.DGLContext.STR2MASK["gpu"]:
            raise ValueError("'device' must be a GPU device.")

        tensor = F.zerocopy_to_dgl_ndarray(tensor)

        transfer_id = _CAPI_DGLAsyncTransfererStartTransfer(self._handle, tensor, ctx)
        return Transfer(transfer_id, self._handle)


_init_api("dgl.dataloading.async_transferer")
