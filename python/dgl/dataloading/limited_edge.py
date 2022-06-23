"""Limited edge dataloader."""
# pylint: disable=bad-super-call, no-value-for-parameter
from typing import Generic
import functools

import numpy as np
import dgl
from dgl.dataloading.dataloader import _TensorizedDatasetIter


class LimitedEdgeDataloader(dgl.dataloading.NodeDataLoader):
    """Limited Edge Dataloader class.

    For the normal dataloader, each batch have a same batch size, which may cause the workload
    unbanlance problem when some batches contains high degree nodes. The limited edge dataloader
    can support different batch sizes among different batches. Users can parse two configures:
    max_node and max_edge, which means the maximum number of nodes in a batch and the maximum
    number of edges in a batch. The limited edge dataloader can provide the largest batch which
    follows the rule.
    """
    def __init__(self, g, nids, sampler, max_node, max_edge, prefix_sum_in_degrees=None, \
        device='cpu', shuffle=False, use_uva=False, drop_last=False, num_workers=0):

        dataset = LimitedEdgeDataset(max_node, max_edge, g, nids, prefix_sum_in_degrees)
        super().__init__(g,
                         dataset,
                         sampler,
                         device=device,
                         use_uva=use_uva,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         use_prefetch_thread=False,
                         num_workers=num_workers)

    def modify_max_edge(self, max_edge):
        """Modify maximum edges."""
        self.dataset.max_edge = max_edge
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_edge = max_edge

    def modify_max_node(self, max_node):
        """Modify maximum nodes."""
        self.dataset.max_node = max_node
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_node = max_node

    def reset_batch_node(self, node_count):
        """Reset batch node."""
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.index -= node_count

    def __setattr__(self, __name, __value):
        super(Generic, self).__setattr__(__name, __value)


class LimitedEdgeDataset(dgl.dataloading.TensorizedDataset):
    """Limited edge tensorized dataset extended from dgl.dataloading."""
    def __init__(self, max_node, max_edge, g, train_nids, prefix_sum_in_degrees=None):
        super().__init__(train_nids, max_node, False)
        self.device = train_nids.device
        self.max_node = max_node
        self.max_edge = max_edge
        # move __iter__ to here
        # TODO: support multi processing
        id_tensor = self._id_tensor[train_nids.to(self._device)]

        self.prefix_sum_in_degrees = prefix_sum_in_degrees
        # Compute the prefix sum in degrees for the graph.
        if self.prefix_sum_in_degrees is None:
            in_degrees = g.in_degrees(train_nids.to(g.device)).cpu()
            prefix_sum_in_degrees = np.cumsum(in_degrees)
            self.prefix_sum_in_degrees = [0]
            self.prefix_sum_in_degrees.extend(prefix_sum_in_degrees.tolist())
            self.prefix_sum_in_degrees.append(2e18)

        self.curr_iter = LimitedEdgeDatasetIter(
            id_tensor, self.max_node, self.max_edge, self.prefix_sum_in_degrees,
            self.drop_last, self._mapping_keys)

    def __getattr__(self, attribute_name):
        if attribute_name in LimitedEdgeDataset.functions:
            function = functools.partial(LimitedEdgeDataset.functions[attribute_name], self)
            return function
        else:
            return super(Generic, self).__getattr__(attribute_name)

    def __iter__(self):
        return self.curr_iter


class LimitedEdgeDatasetIter(_TensorizedDatasetIter):
    """Limited edge tensorized datasetIter."""
    def __init__(self, dataset, max_node, max_edge, prefix_sum_in_degrees, drop_last, mapping_keys):
        super().__init__(dataset, max_node, drop_last, mapping_keys)
        self.max_node = max_node
        self.max_edge = max_edge
        self.prefix_sum_in_degrees = prefix_sum_in_degrees
        self.num_item = self.dataset.shape[0]

    def get_end_idx(self):
        """Get end index by binary search."""
        # binary search
        binary_start = self.index + 1
        binary_end = min(self.index + self.max_node, self.num_item)
        if self.prefix_sum_in_degrees[binary_end] - self.prefix_sum_in_degrees[self.index] \
            < self.max_edge:
            return binary_end
        binary_middle = 0
        while binary_end - binary_start > 1:
            binary_middle = (binary_start + binary_end) // 2
            if self.prefix_sum_in_degrees[binary_middle] - self.prefix_sum_in_degrees[self.index] \
                < self.max_edge:
                binary_start = binary_middle
            else:
                binary_end = binary_middle - 1
        return binary_start

    def _next_indices(self):
        """Overwrite next indices."""
        if self.index >= self.num_item:
            raise StopIteration
        end_idx = self.get_end_idx()
        batch = self.dataset[self.index:end_idx]
        self.index = end_idx
        return batch
