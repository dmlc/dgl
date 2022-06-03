import torch


class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.data = dataloader
    #   self.iter = iter(dataloader)

    def __iter__(self):
        # rewind the iterator for every iteration in for loop
        return iter(self.data)
