"""Base classes and functionalities for feature storages."""

import threading

STORAGE_WRAPPERS = {}


def register_storage_wrapper(type_):
    """Decorator that associates a type to a ``FeatureStorage`` object."""

    def deco(cls):
        STORAGE_WRAPPERS[type_] = cls
        return cls

    return deco


def wrap_storage(storage):
    """Wrap an object into a FeatureStorage as specified by the ``register_storage_wrapper``
    decorators.
    """
    for type_, storage_cls in STORAGE_WRAPPERS.items():
        if isinstance(storage, type_):
            return storage_cls(storage)

    assert isinstance(
        storage, FeatureStorage
    ), "The frame column must be a tensor or a FeatureStorage object, got {}".format(
        type(storage)
    )
    return storage


class _FuncWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, buf, *args):
        buf[0] = self.func(*args)


class ThreadedFuture(object):
    """Wraps a function into a future asynchronously executed by a Python
    ``threading.Thread`.  The function is being executed upon instantiation of
    this object.
    """

    def __init__(self, target, args):
        self.buf = [None]

        thread = threading.Thread(
            target=_FuncWrapper(target),
            args=[self.buf] + list(args),
            daemon=True,
        )
        thread.start()
        self.thread = thread

    def wait(self):
        """Blocks the current thread until the result becomes available and returns it."""
        self.thread.join()
        return self.buf[0]


class FeatureStorage(object):
    """Feature storage object which should support a fetch() operation.  It is the
    counterpart of a tensor for homogeneous graphs, or a dict of tensor for heterogeneous
    graphs where the keys are node/edge types.
    """

    def requires_ddp(self):
        """Whether the FeatureStorage requires the DataLoader to set use_ddp."""
        return False

    def fetch(self, indices, device, pin_memory=False, **kwargs):
        """Retrieve the features at the given indices.

        If :attr:`indices` is a tensor, this is equivalent to

        .. code::

           storage[indices]

        If :attr:`indices` is a dict of tensor, this is equivalent to

        .. code::

           {k: storage[k][indices[k]] for k in indices.keys()}

        The subclasses can choose to utilize or ignore the flag :attr:`pin_memory`
        depending on the underlying framework.
        """
        raise NotImplementedError
