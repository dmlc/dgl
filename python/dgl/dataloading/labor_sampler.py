#
#   Copyright (c) 2022 by Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Based off of neighbor_sampler.py
#

"""Data loading components for labor sampling"""
from ..base import NID, EID
from ..transforms import to_block
from .base import BlockSampler
from ..random import choice
from .. import backend as F


class LaborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    labor sampling for multilayer GNN from
    `(LA)yer-neigh(BOR) Sampling: Defusing Neighborhood Explosion in GNNs
    <https://arxiv.org/abs/2210.13339>`

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type. The neighbors are picked uniformly with default parameters. For every vertex t
    that will be considered to be sampled, there will be a single random variate r_t.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` otherwise, same as :func:`dgl.sampling.sample_neighbors`.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.
    importance_sampling : int, default ``0``
        Whether to use importance sampling or uniform sampling, use of negative values optimizes
        importance sampling probabilities until convergence while use of positive values runs
        optimization steps that many times. If the value is i, then LABOR-i variant is used.
    layer_dependency : bool, default ``False``
        Specifies whether different layers should use same random variates.
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the
        input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to
        the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the
        edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.

    Examples
    --------
    **Node classification**

    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.LaborSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.LaborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform labor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.LaborSampler([5, 10, 15], prob='p')

    **Edge classification and link prediction**

    This class can also work for edge classification and link prediction together
    with :func:`as_edge_prediction_sampler`.

    >>> sampler = dgl.dataloading.LaborSampler([5, 10, 15])
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

    See the documentation :func:`as_edge_prediction_sampler` for more details.

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        importance_sampling=0,
        layer_dependency=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.importance_sampling = importance_sampling
        self.layer_dependency = layer_dependency
        self.set_seed()

    def set_seed(self, random_seed=None):
        '''updates the underlying seed for the sampler'''
        self.random_seed = choice(1e18, 1) if random_seed is None else F.tensor(random_seed)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for i, fanout in enumerate(reversed(self.fanouts)):
            random_seed_i = F.zerocopy_to_dgl_ndarray(
                self.random_seed + (i if not self.layer_dependency else 0)
            )
            frontier, importances = g.sample_labors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                importance_sampling=self.importance_sampling,
                random_seed=random_seed_i,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            eid = frontier.edata[EID]
            block = to_block(
                frontier, seed_nodes, include_dst_in_src=True, src_nodes=None
            )
            block.edata[EID] = eid
            if len(g.canonical_etypes) > 1:
                for etype, importance in zip(
                    g.canonical_etypes, importances
                ):
                    if importance.shape[0] == block.num_edges(etype):
                        block.edata["edge_weights"][etype] = importance
            elif importances[0].shape[0] == block.num_edges():
                block.edata["edge_weights"] = importances[0]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        self.set_seed()
        return seed_nodes, output_nodes, blocks
