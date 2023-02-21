import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from BGNN import BGNNPredictor
from category_encoders import CatBoostEncoder

from dgl.data.utils import load_graphs
from dgl.nn.pytorch import (
    AGNNConv as AGNNConvDGL,
    APPNPConv,
    ChebConv as ChebConvDGL,
    GATConv as GATConvDGL,
    GraphConv,
)
from sklearn import preprocessing
from torch.nn import Dropout, ELU, Linear, ReLU, Sequential


class GNNModelDGL(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.0,
        name="gat",
        residual=True,
        use_mlp=False,
        join_with_mlp=False,
    ):
        super(GNNModelDGL, self).__init__()
        self.name = name
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        self.normalize_input_columns = True
        if name == "gat":
            self.l1 = GATConvDGL(
                in_dim,
                hidden_dim // 8,
                8,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
                activation=F.elu,
            )
            self.l2 = GATConvDGL(
                hidden_dim,
                out_dim,
                1,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=residual,
                activation=None,
            )
        elif name == "gcn":
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, out_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == "cheb":
            self.l1 = ChebConvDGL(in_dim, hidden_dim, k=3)
            self.l2 = ChebConvDGL(hidden_dim, out_dim, k=3)
            self.drop = Dropout(p=dropout)
        elif name == "agnn":
            self.lin1 = Sequential(
                Dropout(p=dropout), Linear(in_dim, hidden_dim), ELU()
            )
            self.l1 = AGNNConvDGL(learn_beta=False)
            self.l2 = AGNNConvDGL(learn_beta=True)
            self.lin2 = Sequential(
                Dropout(p=dropout), Linear(hidden_dim, out_dim), ELU()
            )
        elif name == "appnp":
            self.lin1 = Sequential(
                Dropout(p=dropout),
                Linear(in_dim, hidden_dim),
                ReLU(),
                Dropout(p=dropout),
                Linear(hidden_dim, out_dim),
            )
            self.l1 = APPNPConv(k=10, alpha=0.1, edge_drop=0.0)

    def forward(self, graph, features):
        h = features
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(features)), 1)
            else:
                h = self.mlp(features)
        if self.name == "gat":
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name in ["appnp"]:
            h = self.lin1(h)
            logits = self.l1(graph, h)
        elif self.name == "agnn":
            h = self.lin1(h)
            h = self.l1(graph, h)
            h = self.l2(graph, h)
            logits = self.lin2(h)
        elif self.name == "che3b":
            lambda_max = dgl.laplacian_lambda_max(graph)
            h = self.drop(h)
            h = self.l1(graph, h, lambda_max)
            logits = self.l2(graph, h, lambda_max)
        elif self.name == "gcn":
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)

        return logits


def read_input(input_folder):
    X = pd.read_csv(f"{input_folder}/X.csv")
    y = pd.read_csv(f"{input_folder}/y.csv")

    categorical_columns = []
    if os.path.exists(f"{input_folder}/cat_features.txt"):
        with open(f"{input_folder}/cat_features.txt") as f:
            for line in f:
                if line.strip():
                    categorical_columns.append(line.strip())

    cat_features = None
    if categorical_columns:
        columns = X.columns
        cat_features = np.where(columns.isin(categorical_columns))[0]

        for col in list(columns[cat_features]):
            X[col] = X[col].astype(str)

    gs, _ = load_graphs(f"{input_folder}/graph.dgl")
    graph = gs[0]

    with open(f"{input_folder}/masks.json") as f:
        masks = json.load(f)

    return graph, X, y, cat_features, masks


def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)


def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X


def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(
        A[np.ix_(train_mask, cat_features)], b[train_mask]
    )
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(
        A[np.ix_(val_mask + test_mask, cat_features)]
    )
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)


if __name__ == "__main__":
    # datasets can be found here: https://www.dropbox.com/s/verx1evkykzli88/datasets.zip
    # Read dataset
    input_folder = "datasets/avazu"
    graph, X, y, cat_features, masks = read_input(input_folder)
    train_mask, val_mask, test_mask = (
        masks["0"]["train"],
        masks["0"]["val"],
        masks["0"]["test"],
    )

    encoded_X = X.copy()
    normalizeFeatures = False
    replaceNa = True

    if len(cat_features):
        encoded_X = encode_cat_features(
            encoded_X, y, cat_features, train_mask, val_mask, test_mask
        )
    if normalizeFeatures:
        encoded_X = normalize_features(
            encoded_X, train_mask, val_mask, test_mask
        )
    if replaceNa:
        encoded_X = replace_na(encoded_X, train_mask)

    # specify parameters
    task = "regression"
    hidden_dim = 128
    trees_per_epoch = 5  # 5-10 are good values to try
    backprop_per_epoch = 5  # 5-10 are good values to try
    lr = 0.1  # 0.01-0.1 are good values to try
    append_gbdt_pred = (
        False  # this can be important for performance (try True and False)
    )
    train_input_features = False
    gbdt_depth = 6
    gbdt_lr = 0.1

    out_dim = (
        y.shape[1] if task == "regression" else len(set(y.iloc[test_mask, 0]))
    )
    in_dim = out_dim + X.shape[1] if append_gbdt_pred else out_dim

    # specify GNN model
    gnn_model = GNNModelDGL(in_dim, hidden_dim, out_dim)

    # initialize BGNN model
    bgnn = BGNNPredictor(
        gnn_model,
        task=task,
        loss_fn=None,
        trees_per_epoch=trees_per_epoch,
        backprop_per_epoch=backprop_per_epoch,
        lr=lr,
        append_gbdt_pred=append_gbdt_pred,
        train_input_features=train_input_features,
        gbdt_depth=gbdt_depth,
        gbdt_lr=gbdt_lr,
    )

    # train
    metrics = bgnn.fit(
        graph,
        encoded_X,
        y,
        train_mask,
        val_mask,
        test_mask,
        original_X=X,
        cat_features=cat_features,
        num_epochs=100,
        patience=10,
        metric_name="loss",
    )

    bgnn.plot_interactive(
        metrics,
        legend=["train", "valid", "test"],
        title="Avazu",
        metric_name="loss",
    )
