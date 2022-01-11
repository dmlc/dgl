import asyncio
import threading
# Solution to enable running asyncio.run in Jupyter notebooks.
# Reference: https://stackoverflow.com/a/69514930

def _start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AsyncIO(metaclass=Singleton):
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=_start_background_loop,
            args=(self._loop,),
            daemon=True)
        loop_thread.start()
        self._loop_thread = loop_thread

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def gather(self, *futures, return_exceptions=False):
        return asyncio.gather(*futures, loop=self._loop, return_exceptions=return_exceptions)
