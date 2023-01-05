#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from collections.abc import Sequence
from ...base import NID, EID, DGLError
from ...dataloading import GraphCollator

class GraphSource:
    def fetch_graph(self, ids, output_device):
        """
        Get the graph(s) corresponding to the given set of ids.

        Parameters
        ----------
        ids : Tensor or int
            If this graphsource supports dynamic batching, a tensor of ids to
            include in the batch should be given.

            If this graphsource does not support dynamic batching, then a
            single id should be given, and it should correspond to the batch
            number rather than the items in the batch.

        output_device : torch.device
            The device the graphs should be returned on.

        Returns
        -------
        DGLHeterograph or List of DGLHeterographs/DGLBlocks
            For graph convolution models, this should return the MFG graphs as
            a list, with the outermost (input) graph at position 0, and the
            innermost (output) graph at the last position.

            For other models this should just return the graph corresponding to
            the mini-batch.
        """
        pass


class SampledGraphSource:
    def __init__(self, g, sampler):
        self._g = g
        self._sampler = sampler

    def fetch_graph(self, ids, output_device):
        self._sampler.output_device = output_device
        batch = self._sampler.sample(
            self._g, ids, copy_ndata=False, copy_edata=False)
        if len(batch) == 3:
            # node prediction
            _input_nodes, _output_nodes, blocks = batch
            return blocks
        else:
            # edge prediction
            assert len(batch) == 4, "Unexpected batch size: {}".format(len(batch))
            _input_nodes, pair_graph, neg_pair_graph, blocks = batch
            return pair_graph, neg_pair_graph, blocks

class BatchedGraphSource:
    def __init__(self, graphs):
        self._graphs = graphs

    def __len__(self):
        return len(self._graphs)

    def fetch_graph(self, idx, output_device):
        graphs = self._graphs[idx]
        if not isinstance(graphs, Sequence):
            graphs = [graphs]
        return [graph.to(output_device) for graph in graphs]

class DatasetGraphSource:
    def __init__(self, dataset):
        self._dataset = dataset
        self._collator = GraphCollator()

    def __len__(self):
        return len(self._dataset)

    def fetch_graph(self, ids, output_device):
        graphs = [self._dataset[i] for i in ids]
        batched_graphs = self._collator.collate(graphs)
        # copy graph (and graph features) to device
        return [g.to(output_device) for g in batched_graphs]


