"""Wraps a GraphBolt DataPipe in a daemonic thread."""

import atexit
import threading
from queue import Empty, Full, Queue

import torch
from torchdata.datapipes.iter import IterDataPipe

from ..utils.exception import ExceptionWrapper

PYTHON_EXIT_STATUS = False


def _set_python_exit_flag():
    global PYTHON_EXIT_STATUS
    PYTHON_EXIT_STATUS = True


atexit.register(_set_python_exit_flag)


def _put_if_event_not_set(queue, result, event):
    while not event.is_set():
        try:
            queue.put(result, timeout=1.0)
            break
        except Full:
            continue


class ThreadWrapper(IterDataPipe):
    """Wraps a DataPipe inside a daemonic thread.

    The given DataPipe will be executed in an individual daemonic thread,
    which puts the produced results in a queue. Iterating over this DataPipe
    object yields elements from the queue instead of yielding directly
    from the DataPipe. This achieves prefetching with Python threads.

    Best used for IO-bound DataPipes.

    Parameters
    ----------
    dp : DataPipe
        The DataPipe object.
    torch_num_threads : int, optional
        The number of PyTorch threads to set.
    buffer_size : int, optional
        The prefetch queue size.
    timeout : int, optional
        Prefetch thread timeout.
    """

    def __init__(
        self,
        dp,
        torch_num_threads=1,
        buffer_size=1,
        timeout=30,
    ):
        self.queue = Queue(buffer_size)
        self.dp = dp
        self.torch_num_threads = torch_num_threads
        self.done_event = threading.Event()
        self.timeout = timeout
        self.thread = threading.Thread(
            target=self._thread_entry,
            daemon=True,
        )
        self.thread.start()
        self._shutting_down = False

    def _thread_entry(self):
        if self.torch_num_threads is not None:
            torch.set_num_threads(self.torch_num_threads)
        it = iter(self.dp)
        try:
            while not self.done_event.is_set():
                try:
                    item = next(it)
                except StopIteration:
                    break
                _put_if_event_not_set(self.queue, (item, None), self.done_event)
            _put_if_event_not_set(self.queue, (None, None), self.done_event)
        except:  # pylint: disable=bare-except
            _put_if_event_not_set(
                self.queue,
                (None, ExceptionWrapper(where="in prefetcher")),
                self.done_event,
            )

    def __iter__(self):
        while True:
            try:
                item, exception = self.queue.get(timeout=self.timeout)
            except Empty:
                raise RuntimeError(
                    f"Prefetcher timeout at {self.timeout} seconds."
                )
            if item is None:
                self.thread.join()
                if exception is None:
                    return
                exception.reraise()
            yield item

    def _shutdown(self):
        if PYTHON_EXIT_STATUS is True or PYTHON_EXIT_STATUS is None:
            return
        if not self._shutting_down:
            try:
                self._shutting_down = True
                self.done_event.set()

                try:
                    self.queue.get_nowait()
                except:  # pylint: disable=bare-except
                    pass
                self.thread.join()
            except:  # pylint: disable=bare-except
                pass

    def __del__(self):
        self._shutdown()
