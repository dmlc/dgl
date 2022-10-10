"""Best hyperparameters found."""
import torch

MWE_GCN_proteins = {
    "num_ew_channels": 8,
    "num_epochs": 2000,
    "in_feats": 1,
    "hidden_feats": 10,
    "out_feats": 112,
    "n_layers": 3,
    "lr": 2e-2,
    "weight_decay": 0,
    "patience": 1000,
    "dropout": 0.2,
    "aggr_mode": "sum",  ## 'sum' or 'concat' for the aggregation across channels
    "ewnorm": "both",
}

MWE_DGCN_proteins = {
    "num_ew_channels": 8,
    "num_epochs": 2000,
    "in_feats": 1,
    "hidden_feats": 10,
    "out_feats": 112,
    "n_layers": 2,
    "lr": 1e-2,
    "weight_decay": 0,
    "patience": 300,
    "dropout": 0.5,
    "aggr_mode": "sum",
    "residual": True,
    "ewnorm": "none",
}


def get_exp_configure(args):
    if args["model"] == "MWE-GCN":
        return MWE_GCN_proteins
    elif args["model"] == "MWE-DGCN":
        return MWE_DGCN_proteins
