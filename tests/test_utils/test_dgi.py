import dgl
import unittest

@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')

def test_dgi():
    import torch
    import torch.nn.functional as F
    from dgl.utils.dgi import dgl_symbolic_trace, split_module, InferenceHelper

    class GCN(torch.nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
            self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
            self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
            self.n_layers = 3

        def forward(self, graph, x0):
            x1 = F.relu(self.conv1(graph, x0))
            x2 = F.relu(self.conv2(graph, x1))
            x3 = F.relu(self.conv3(graph, x2))
            return x3

    # test trace
    model = GCN(5, 5, 5)
    traced = dgl_symbolic_trace(model)

    # test split
    splitted = split_module(traced)
    assert splitted is not None

    # test inference
    g = dgl.graph(([1, 2, 3, 4, 0, 3, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6]))
    feat = torch.zeros((7, 5))
    helper = InferenceHelper(model, (), 2, torch.device("cpu"))
    helper_pred = helper.inference(g, feat)
    assert helper_pred.shape == (7, 5)

if __name__ == "__main__":
    test_dgi()
