##
#   Copyright 2021 Contributors 
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

import dgl
import backend as F
import unittest
from dgl.contrib import MultiGPUNodeDataLoader

class DummyCommunicator:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def sparse_all_to_all_pull(self, req_idx, value, partition):
        # assume all indices are local
        idxs = partition.map_to_local(req_idx)
        return F.gather_row(value, idxs)

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_node_dataloader():
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]))
    g1.ndata['feat'] = F.copy_to(F.randn((5, 8)), F.cpu())
    graph_data_dim = g1.ndata['feat'].shape[1]
    node_feat = F.copy_to(F.randn((5, 4)), F.cpu())
    node_label = F.randint(low=0, high=3, shape=[5], dtype=F.int32, ctx=F.cpu())

    batch_size = 2

    dataloader = MultiGPUNodeDataLoader(
        g1, g1.nodes(), sampler, device=F.ctx(),
        node_feat=node_feat,
        node_label=node_label,
        use_ddp=False,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    for input_nodes, output_nodes, blocks, block_feat, block_label in dataloader:
        # make sure everything is on the GPU
        assert input_nodes.device == F.ctx()
        assert output_nodes.device == F.ctx()
        for block in blocks:
            assert block.device == F.ctx()
        assert block_feat.device == F.ctx()
        assert block_label.device == F.ctx()

        # make sure we have the same features
        for block in blocks:
            for ntype in block.ntypes:
                block_data_dim = block.ndata['feat'][ntype].shape[1]
                assert block_data_dim == graph_data_dim
        exp_feat = node_feat[input_nodes]
        act_feat = F.copy_to(block_feat, F.cpu())
        assert F.array_equal(exp_feat, act_feat)
        exp_label = node_label[output_nodes]
        act_label = F.copy_to(block_label, F.cpu())
        assert F.array_equal(exp_label, act_label)


@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_node_dataloader_hg():

    g1 = dgl.heterograph({
         ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0]),
         ('user', 'followed-by', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2]),
         ('user', 'play', 'game'): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
         ('game', 'played-by', 'user'): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5])
    })
    graph_data_dim = {}
    for ntype in g1.ntypes:
        g1.nodes[ntype].data['feat'] = F.copy_to(F.randn((g1.num_nodes(ntype), 8)), F.cpu())

        graph_data_dim[ntype] = g1.nodes[ntype].data['feat'].shape[1]

    node_feat = {}
    for ntype in g1.ntypes:
        node_feat[ntype] = F.copy_to(F.randn((g1.num_nodes(ntype), 16)), F.cpu())

    batch_size = max(g1.num_nodes(nty) for nty in g1.ntypes)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = MultiGPUNodeDataLoader(
        g1, {nty: g1.nodes(nty) for nty in g1.ntypes},
        sampler, device=F.ctx(),
        node_feat=node_feat,
        use_ddp=False,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    for input_nodes, output_nodes, blocks, block_feat in dataloader:
        # make sure everything is on the GPU
        for _, tensor in input_nodes.items():
            assert tensor.device == F.ctx()
        for _, tensor in output_nodes.items():
            assert tensor.device == F.ctx()
        for block in blocks:
            assert block.device == F.ctx()
        for _, feat in block_feat.items():
            assert feat.device == F.ctx()

        # make sure we have the same features
        for block in blocks:
            for ntype in block.ntypes:
                print("block.ndata['feat'] = {}".format(block.ndata['feat']))
                print("ntype = {}".format(ntype))
                block_data_dim = block.ndata['feat'][ntype].shape[1]
                assert block_data_dim == graph_data_dim[ntype]
        for ntype in g1.ntypes:
            exp_feat = node_feat[ntype][input_nodes[ntype]]
            act_feat = F.copy_to(block_feat[ntype], F.cpu())
            assert F.array_equal(exp_feat, act_feat)

if __name__ == '__main__':
    test_node_dataloader()
    test_node_dataloader_hg()
