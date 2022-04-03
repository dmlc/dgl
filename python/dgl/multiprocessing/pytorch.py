"""PyTorch multiprocessing wrapper."""
from functools import wraps
import os
from collections import namedtuple
import traceback
from _thread import start_new_thread
import torch.multiprocessing as mp

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
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        target = thread_wrapped_func(target)
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

ProcessContext = namedtuple('ProcessContext', ['queue', 'barrier', 'rank', 'nprocs'])
mp_timeout = int(os.environ.get('DGL_MP_TIMEOUT', '10'))
_PROCESS_CONTEXT = None

def call_once_and_share(func, rank=0):
    """Invoke the function in a single process of the process group spawned by
    :func:`spawn`, and share the result to other processes.

    Requires the subprocesses to be spawned with :func:`dgl.multiprocessing.pytorch.spawn`.

    Parameters
    ----------
    func : callable
        Any callable that accepts no arguments and returns an arbitrary object.
    rank : int, optional
        The process ID to actually execute the function.
    """
    global _PROCESS_CONTEXT
    if _PROCESS_CONTEXT is None:
        raise RuntimeError(
            'call_once_and_share can only be called within processes spawned by '
            'dgl.multiprocessing.spawn() function. '
            'Please replace torch.multiprocessing.spawn() with dgl.multiprocessing.spawn().')

    if _PROCESS_CONTEXT.rank == rank:
        result = func()
        for _ in range(_PROCESS_CONTEXT.nprocs - 1):
            _PROCESS_CONTEXT.queue.put(result)
    else:
        result = _PROCESS_CONTEXT.queue.get(timeout=mp_timeout)
    _PROCESS_CONTEXT.barrier.wait(timeout=mp_timeout)
    return result

def _spawn_entry(rank, queue, barrier, nprocs, fn, *args):
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = ProcessContext(queue, barrier, rank, nprocs)
    fn(rank, *args)

def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    """A wrapper around :func:`torch.multiprocessing.spawn` that allows calling
    DGL-specific multiprocessing functions in :mod:`dgl.multiprocessing` namespace."""
    ctx = mp.get_context(start_method)

    # The following two queues are for call_once_and_share
    queue = ctx.Queue()
    barrier = ctx.Barrier(nprocs)

    mp.spawn(_spawn_entry, args=(queue, barrier, nprocs, fn) + tuple(args), nprocs=nprocs,
             join=join, daemon=daemon, start_method=start_method)
