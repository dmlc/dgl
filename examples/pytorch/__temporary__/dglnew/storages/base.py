import asyncio
from functools import partial

class FeatureStorage(object):
    def fetch(self, indices, device, pin_memory=False):
        raise NotImplementedError

    async def async_fetch(self, indices, device, pin_memory=False):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, partial(self.fetch, indices, device, pin_memory))
        return result
