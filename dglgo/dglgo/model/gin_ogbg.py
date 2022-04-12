import torch.nn as nn
from dgl.nn import GINEConv

class OGBGGIN(nn.Module):
    def __init__(self,
                 data_info: dict):
        """Graph Isomorphism Network (GIN) variant introduced in baselines
        for OGB graph property prediction datasets

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        """
        super(OGBGGIN, self).__init__()
        self.data_info = data_info

    def forward(self, graph, node_feat, edge_feat):
        return NotImplementedError
