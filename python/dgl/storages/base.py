import threading
from functools import partial


storage_wrappers = {}
def register_storage_wrapper(type_):
    def deco(cls):
        storage_wrappers[type_] = cls
        return cls
    return deco

class _FuncWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, buf, *args):
        buf[0] = self.func(*args)

class ThreadFutureWrapper(object):
    def __init__(self, target, args):
        self.buf = [None]

        thread = threading.Thread(
            target=_FuncWrapper(target),
            args=[self.buf] + list(args),
            daemon=True)
        thread.start()
        self.thread = thread

    def wait(self):
        self.thread.join()
        return self.buf[0]

class FeatureStorage(object):
    """Feature storage object which should support a fetch() operation.  It is the
    counterpart of a tensor for homogeneous graphs, or a dict of tensor for heterogeneous
    graphs where the keys are node/edge types.
    """
    def requires_ddp(self):
        """Whether the FeatureStorage requires the DataLoader to set use_ddp.
        """
        return False

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
