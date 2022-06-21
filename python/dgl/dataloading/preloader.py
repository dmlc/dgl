##
#   Copyright 2021-2022 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

from . import DataLoader
from queue import Queue
import threading
import torch

__all__ = ['preload_with_thread']

def _preloader_entry(num_threads, it, queue, done_event):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    try:
        batch = None
        while not done_event.is_set():
            try:
                batch = next(it)
            except StopIteration:
                break
            queue.put((batch, None))
        queue.put((None, None))
    except Exception as e:
        queue.put((None, e))

class _ThreadedPreloadingIterator:
    def __init__(self, super_iter, use_worker_threads):
        self._shutting_down = False
        self._iter = super_iter
        self._queue = Queue(1)
        self._num_threads = torch.get_num_threads() if use_worker_threads \
            else None
        self._done_event = threading.Event()
        self._thread = threading.Thread(
            target=_preloader_entry, \
            args=(self._num_threads, self._iter, self._queue, \
                self._done_event), \
            daemon=True)

        self._thread.start()

    def __next__(self):
        batch, exception = self._queue.get()
        if batch is None:
            self._thread.join()
            if exception is None:
                raise StopIteration
            exception.reraise()
        return batch

    def _shutdown(self):
        if not self._shutting_down:
            try:
                self._shutting_down = True
                self._done_event.set()

                try:
                    self._queue.get_nowait()     # In case the thread is blocking on put().
                except:     # pylint: disable=bare-except
                    pass

                self._thread.join()
            except:         # pylint: disable=bare-except
                pass

    def __del__(self):
        self._shutdown()

class _IterableWrapper:
    def __init__(self, item, wrapper):
        self._item = item
        self._wrapper = wrapper

    def __iter__(self):
        return self._wrapper(iter(self._item), \
                             use_worker_threads=self._item.num_workers > 0 )


def preload_with_thread(dataloader):
    """ Use a separate thread to perform __next__ on the dataloader object.
    """
    return _IterableWrapper(dataloader, _ThreadedPreloadingIterator)
