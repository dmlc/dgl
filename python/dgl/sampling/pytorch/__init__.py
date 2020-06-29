"""DGL PyTorch DataLoaders"""
from torch.utils.data import DataLoader
from ..dataloader import NodeCollator

class NodeDataLoader(DataLoader):
    """PyTorch dataloader for batch-iterating over a set of nodes, generating the list
    of blocks as computation dependency of the said minibatch.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : :py:class:`~dgl.sampling.BlockSampler`
        The neighborhood sampler.
    kwargs : dict
        Arguments being passed to `torch.utils.data.DataLoader`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> dataloader = dgl.sampling.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)
    """
    def __init__(self, g, nids, block_sampler, **kwargs):
        self.collator = NodeCollator(g, nids, block_sampler)
        super().__init__(self.collator.dataset, collate_fn=self.collator.collate, **kwargs)
