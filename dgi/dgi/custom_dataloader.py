from typing import Generic
import functools

import torch
import dgl
from dgl.dataloading.dataloader import _TensorizedDatasetIter


def _divide_by_worker(dataset):
    num_samples = dataset.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        num_samples_per_worker = num_samples // worker_info.num_workers + 1
        start = num_samples_per_worker * worker_info.id
        end = min(start + num_samples_per_worker, num_samples)
        dataset = dataset[start:end]
    return dataset

class CustomDataloader(dgl.dataloading.NodeDataLoader):
    def __init__(self, g, nids, sampler, start_max_node=1000, start_max_edge=10000, prefix_sum_in_degrees=None, \
        device='cpu', shuffle=False, use_uva=False, num_workers=0):

        custom_dataset = CustomDataset(start_max_node, start_max_edge, g, nids, prefix_sum_in_degrees)
        super().__init__(g,
                         custom_dataset,
                         sampler,
                         device=device,
                         use_uva=use_uva,
                         shuffle=shuffle,
                         drop_last=False,
                         use_prefetch_thread=False,
                         num_workers=num_workers)

    def modify_max_edge(self, max_edge):
        self.dataset.max_edge = max_edge
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_edge = max_edge

    def modify_max_node(self, max_node):
        self.dataset.max_node = max_node
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_node = max_node

    def reset_batch_node(self, node_count):
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.index -= node_count

    def __setattr__(self, __name, __value) -> None:
        return super(Generic, self).__setattr__(__name, __value)

class CustomDataset(dgl.dataloading.TensorizedDataset):
    def __init__(self, max_node, max_edge, g, train_nids, prefix_sum_in_degrees=None):
        super().__init__(train_nids, max_node, False)
        self.device = train_nids.device
        self.max_node = max_node
        self.max_edge = max_edge
        # move __iter__ to here
        # TODO not support multi processing yet
        # indices = _divide_by_worker(train_nids)
        id_tensor = self._id_tensor[train_nids.to(self._device)]
        self.prefix_sum_in_degrees = prefix_sum_in_degrees
        if self.prefix_sum_in_degrees is None:
            in_degrees = g.in_degrees(train_nids.to(g.device))
            self.prefix_sum_in_degrees = [0]
            self.prefix_sum_in_degrees.extend(in_degrees.tolist())
            for i in range(1, len(self.in_degrees)):
                self.prefix_sum_in_degrees[i] += self.prefix_sum_in_degrees[i - 1]
            self.prefix_sum_in_degrees.append(2e18)
        self.curr_iter = CustomDatasetIter(
            id_tensor, self.max_node, self.max_edge, self.prefix_sum_in_degrees, self.drop_last, self._mapping_keys)

    def __getattr__(self, attribute_name):
        if attribute_name in CustomDataset.functions:
            function = functools.partial(CustomDataset.functions[attribute_name], self)
            return function
        else:
            return super(Generic, self).__getattr__(attribute_name)

    def __iter__(self):
        return self.curr_iter

class CustomDatasetIter(_TensorizedDatasetIter):
    def __init__(self, dataset, max_node, max_edge, prefix_sum_in_degrees, drop_last, mapping_keys):
        super().__init__(dataset, max_node, drop_last, mapping_keys)
        self.max_node = max_node
        self.max_edge = max_edge
        self.prefix_sum_in_degrees = prefix_sum_in_degrees
        self.num_item = self.dataset.shape[0]

    def get_end_idx(self):
        # binary search
        binary_start = self.index + 1
        binary_end = min(self.index + self.max_node, self.num_item)
        if self.prefix_sum_in_degrees[binary_end] - self.prefix_sum_in_degrees[self.index] < self.max_edge:
            return binary_end
        binary_middle = 0
        while binary_end - binary_start > 5:
            binary_middle = (binary_start + binary_end) // 2
            if self.prefix_sum_in_degrees[binary_middle] - self.prefix_sum_in_degrees[self.index] < self.max_edge:
                binary_start = binary_middle
            else:
                binary_end = binary_middle - 1
        return binary_middle

    def _next_indices(self):
        if self.index >= self.num_item:
            raise StopIteration
        end_idx = self.get_end_idx()
        batch = self.dataset[self.index:end_idx]
        self.index = end_idx
        return batch
