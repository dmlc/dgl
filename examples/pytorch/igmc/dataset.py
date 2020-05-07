import os
import random

class MovieLensDataset(object):
    def __init__(self, subgraphs):
        self.subgraphs = subgraphs
        order = list(range(len(self.subgraphs)))
    
    def __call__(self,  mode='train', batch_size=32, k=1):
        if mode == 'train':
            random.shuffle(order)
        
        batch = []
        for idx in order:
            batch.append(self.subgraphs[idx])
            if len(batch) == batch_size:
                yield batch
                batch = []

        if len(batch) != 0:
            yield batch

