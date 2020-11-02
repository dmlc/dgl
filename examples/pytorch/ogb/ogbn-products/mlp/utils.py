import torch


class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:
            raise StopIteration() from None


class BatchSampler(object):
    def __init__(self, n, batch_size, shuffle=False):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle:
                perm = torch.randperm(self.n)
            else:
                perm = torch.arange(start=0, end=self.n)
            shuf = perm.split(self.batch_size)
            for shuf_batch in shuf:
                yield shuf_batch
            yield None
