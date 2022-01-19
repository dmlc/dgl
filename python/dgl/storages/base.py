import asyncio
from functools import partial

class FeatureStorage(object):
    """Feature storage object which should support a fetch() operation.  It is the
    counterpart of a tensor for homogeneous graphs, or a dict of tensor for heterogeneous
    graphs where the keys are node/edge types.
    """
    def fetch(self, indices, device, pin_memory=False):
        """Retrieve the features at the given indices.

        If :attr:`indices` is a tensor, this is equivalent to

        .. code::

           storage[indices]

        If :attr:`indices` is a dict of tensor, this is equivalent to

        .. code::

           {k: storage[k][indices[k]] for k in indices.keys()}
        """
        raise NotImplementedError

    async def async_fetch(self, indices, device, pin_memory=False):
        """Default implementation of fetching features asynchronously with asyncio.

        Will be invoked if the DataLoader's ``use_asyncio`` flag is True.

        You can implement a synchronous ``fetch`` method and call this function with
        asyncio to make your fetches asynchronous.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, partial(self.fetch, indices, device, pin_memory))
        return result
