from . import DataLoader


class PrefetchingIterator():
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.queue = Queue(1)

    def __iter__(self):
        thread = threading.Thread(
            target=_prefetcher_entry,
            args=(self, self.num_threads

