"""Graph Bolt DataLoaders"""

import torch.utils.data


class SingleProcessDataLoader(torch.utils.data.DataLoader):
    """Single process DataLoader.

    Iterates over the data pipeline in the main process.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    """

    # In the single process dataloader case, we don't need to do any
    # modifications to the datapipe, and we just PyTorch's native
    # dataloader as-is.
    #
    # The exception is that batch_size should be None, since we already
    # have minibatch sampling and collating in MinibatchSampler.
    def __init__(self, datapipe):
        super().__init__(datapipe, batch_size=None, num_workers=0)
