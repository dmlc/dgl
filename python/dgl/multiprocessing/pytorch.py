"""PyTorch multiprocessing wrapper."""
from functools import wraps
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
