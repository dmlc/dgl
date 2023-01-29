"""PyTorch multiprocessing wrapper."""
import random
import traceback
from _thread import start_new_thread
from functools import wraps

import torch
import torch.multiprocessing as mp

from ..utils import create_shared_mem_array, get_shared_mem_array


def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorated_function


# pylint: disable=missing-docstring
class Process(mp.Process):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        *,
        daemon=None
    ):
        target = thread_wrapped_func(target)
        super().__init__(group, target, name, args, kwargs, daemon=daemon)


def _get_shared_mem_name(id_):
    return "shared" + str(id_)


def call_once_and_share(func, shape, dtype, rank=0):
    """Invoke the function in a single process of the PyTorch distributed process group,
    and share the result with other processes.

    Parameters
    ----------
    func : callable
        Any callable that accepts no arguments and returns an arbitrary object.
    shape : tuple[int]
        The shape of the shared tensor.  Must match the output of :attr:`func`.
    dtype : torch.dtype
        The data type of the shared tensor.  Must match the output of :attr:`func`.
    rank : int, optional
        The process ID to actually execute the function.
    """
    current_rank = torch.distributed.get_rank()
    dist_buf = torch.LongTensor([1])

    if torch.distributed.get_backend() == "nccl":
        # Use .cuda() to transfer it to the correct device.  Should be OK since
        # PyTorch recommends the users to call set_device() after getting inside
        # torch.multiprocessing.spawn()
        dist_buf = dist_buf.cuda()

    # Process with the given rank creates and populates the shared memory array.
    if current_rank == rank:
        # PyTorch Lightning 1.6+ seems to set the random seed during process spawning
        # to the same seed value.
        random_ = random.Random()
        id_ = random_.getrandbits(32)
        name = _get_shared_mem_name(id_)
        result = create_shared_mem_array(name, shape, dtype)
        result[:] = func()
        dist_buf[0] = id_

    # Broadcasts the name of the shared array to other processes.
    torch.distributed.broadcast(dist_buf, rank)
    # If no exceptions, other processes open the same shared memory object.
    if current_rank != rank:
        id_ = dist_buf.item()
        name = _get_shared_mem_name(id_)
        result = get_shared_mem_array(name, shape, dtype)

    return result


def shared_tensor(shape, dtype=torch.float32):
    """Create a tensor in shared memory accessible by all processes within the same
    ``torch.distributed`` process group.

    The content is uninitialized.

    Parameters
    ----------
    shape : tuple[int]
        The shape of the tensor.
    dtype : torch.dtype, optional
        The dtype of the tensor.

    Returns
    -------
    Tensor
        The shared tensor.
    """
    return call_once_and_share(
        lambda: torch.empty(*shape, dtype=dtype), shape, dtype
    )
