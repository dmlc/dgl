import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import dgl
import numpy as np


def collate_fn(batch):
    """
    collate_fn for dataset batching
    transform ndata to tensor (in gpu is available)
    """
    graphs, labels = map(list, zip(*batch))
    #cuda = torch.cuda.is_available()

    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = value.float()
    batched_graphs = dgl.batch(graphs)

    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))

    return batched_graphs, batched_labels


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(GraphDataLoader, self).__init__(dataset, batch_size, shuffle,
                                              collate_fn=collate_fn, **kwargs)
