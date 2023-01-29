import torch as th
import torch.nn.functional as F

GCN_CONFIG = {
    "extra_args": [16, 1, F.relu, 0.5],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}

GAT_CONFIG = {
    "extra_args": [8, 1, [8] * 1 + [1], F.elu, 0.6, 0.6, 0.2, False],
    "lr": 0.005,
    "weight_decay": 5e-4,
}

GRAPHSAGE_CONFIG = {
    "extra_args": [16, 1, F.relu, 0.5, "gcn"],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}

APPNP_CONFIG = {
    "extra_args": [64, 1, F.relu, 0.5, 0.5, 0.1, 10],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}

TAGCN_CONFIG = {
    "extra_args": [16, 1, F.relu, 0.5],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}

AGNN_CONFIG = {
    "extra_args": [32, 2, 1.0, True, 0.5],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}

SGC_CONFIG = {
    "extra_args": [None, 2, False],
    "lr": 0.2,
    "weight_decay": 5e-6,
}

GIN_CONFIG = {
    "extra_args": [16, 1, 0, True],
    "lr": 1e-2,
    "weight_decay": 5e-6,
}

CHEBNET_CONFIG = {
    "extra_args": [32, 1, 2, True],
    "lr": 1e-2,
    "weight_decay": 5e-4,
}
