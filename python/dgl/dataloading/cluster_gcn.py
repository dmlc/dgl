"""Cluster-GCN samplers."""
import os
import pickle

import numpy as np

from .. import backend as F
from ..base import DGLError
from ..partition import metis_partition_assignment
from .base import Sampler, set_edge_lazy_features, set_node_lazy_features


class ClusterGCNSampler(Sampler):
    """Cluster sampler from `Cluster-GCN: An Efficient Algorithm for Training
    Deep and Large Graph Convolutional Networks
    <https://arxiv.org/abs/1905.07953>`__

    This sampler first partitions the graph with METIS partitioning, then it caches the nodes of
    each partition to a file within the given cache directory.

    The sampler then selects the graph partitions according to the provided
    partition IDs, take the union of all nodes in those partitions, and return an
    induced subgraph in its :attr:`sample` method.

    Parameters
    ----------
    g : DGLGraph
        The original graph.  Must be homogeneous and on CPU.
    k : int
        The number of partitions.
    cache_path : str
        The path to the cache directory for storing the partition result.
    balance_ntypes, balkance_edges, mode :
        Passed to :func:`dgl.metis_partition_assignment`.
    prefetch_ndata : list[str], optional
        The node data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    prefetch_edata : list[str], optional
        The edge data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of partition indices.

    Examples
    --------
    **Node classification**

    With this sampler, the data loader will accept the list of partition IDs as
    indices to iterate over.  For instance, the following code first splits the
    graph into 1000 partitions using METIS, and at each iteration it gets a subgraph
    induced by the nodes covered by 20 randomly selected partitions.

    >>> num_parts = 1000
    >>> sampler = dgl.dataloading.ClusterGCNSampler(g, num_parts)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, torch.arange(num_parts), sampler,
    ...     batch_size=20, shuffle=True, drop_last=False, num_workers=4)
    >>> for subg in dataloader:
    ...     train_on(subg)
    """

    def __init__(
        self,
        g,
        k,
        cache_path="cluster_gcn.pkl",
        balance_ntypes=None,
        balance_edges=False,
        mode="k-way",
        prefetch_ndata=None,
        prefetch_edata=None,
        output_device=None,
    ):
        super().__init__()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    (
                        self.partition_offset,
                        self.partition_node_ids,
                    ) = pickle.load(f)
            except (EOFError, TypeError, ValueError):
                raise DGLError(
                    f"The contents in the cache file {cache_path} is invalid. "
                    f"Please remove the cache file {cache_path} or specify another path."
                )
            if len(self.partition_offset) != k + 1:
                raise DGLError(
                    f"Number of partitions in the cache does not match the value of k. "
                    f"Please remove the cache file {cache_path} or specify another path."
                )
            if len(self.partition_node_ids) != g.num_nodes():
                raise DGLError(
                    f"Number of nodes in the cache does not match the given graph. "
                    f"Please remove the cache file {cache_path} or specify another path."
                )
        else:
            partition_ids = metis_partition_assignment(
                g,
                k,
                balance_ntypes=balance_ntypes,
                balance_edges=balance_edges,
                mode=mode,
            )
            partition_ids = F.asnumpy(partition_ids)
            partition_node_ids = np.argsort(partition_ids)
            partition_size = F.zerocopy_from_numpy(
                np.bincount(partition_ids, minlength=k)
            )
            partition_offset = F.zerocopy_from_numpy(
                np.insert(np.cumsum(partition_size), 0, 0)
            )
            partition_node_ids = F.zerocopy_from_numpy(partition_node_ids)
            with open(cache_path, "wb") as f:
                pickle.dump((partition_offset, partition_node_ids), f)
            self.partition_offset = partition_offset
            self.partition_node_ids = partition_node_ids

        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device

    def sample(self, g, partition_ids):  # pylint: disable=arguments-differ
        """Sampling function.

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        partition_ids : Tensor
            A 1-D integer tensor of partition IDs.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        node_ids = F.cat(
            [
                self.partition_node_ids[
                    self.partition_offset[i] : self.partition_offset[i + 1]
                ]
                for i in F.asnumpy(partition_ids)
            ],
            0,
        )
        sg = g.subgraph(
            node_ids, relabel_nodes=True, output_device=self.output_device
        )
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg
