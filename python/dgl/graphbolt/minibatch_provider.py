"""Minibatch Loader"""

import torch
from torch.utils.data import functional_datapipe

from .minibatch_transformer import MiniBatchTransformer

__all__ = [
    "MinibatchLoader",
]


@functional_datapipe("load_minibatch")
class MinibatchLoader(MiniBatchTransformer):
    def __init__(self, datapipe, subgraph_dir: str):
        self._subgraph_dir = subgraph_dir
        datapipe = datapipe.transform(self._load_minibatch)
        super().__init__(datapipe)

    def _load_minibatch(self, minibatch):
        torch.cuda.synchronize()
        nid = minibatch.seeds.item()
        seeds, input_nodes, labels, blocks = torch.load(
            f"{self._subgraph_dir}/train-{nid}.pt"
        )
        minibatch.seeds = seeds
        minibatch.input_nodes = input_nodes
        minibatch.labels = labels
        minibatch._blocks = blocks
        return minibatch
