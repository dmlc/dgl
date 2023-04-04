import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, MaxPooling

# source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            MaxPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h, graph=True):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0

        if graph:
            # perform graph sum pooling over all nodes in each layer
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

            return score_over_layer
        else:
            return hidden_rep[-1]


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("attr")
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(train_loader, val_loader, device, model, epochs=350, lr=0.005):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("attr")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        choices=["MUTAG"],
        help="name of dataset (default: MUTAG)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    device = "cpu"

    # load and split dataset
    dataset = GINDataset(
        args.dataset, self_loop=True, degree_as_nlabel=False
    )  # add self_loop and disable one-hot encoding for input features

    labels = [l for _, l in dataset]

    train_idx, val_idx = split_fold10(labels)

    # create dataloader
    train_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )

    # create GIN model
    in_size = dataset.dim_nfeats
    out_size = dataset.gclasses
    hidden_size = 128

    model = GIN(in_size, hidden_size, out_size).to(device)

    """
    # model training/validating
    print("Training...")
    train(train_loader, val_loader, device, model)

    print("Evaluating...")
    print(f"acc: {round(evaluate(train_loader, device, model), 2)}")

    torch.save(model, 'model.dt')
    """

    model = torch.load('model.dt')
    model.eval()
    ##########
    # import pickle as pkl
    # with open(r'C:\Users\kunmu\Downloads\Mutagenicity.pkl\Mutagenicity.pkl', 'rb') as fin:
    #     original_adjs, original_features, original_labels = pkl.load(fin)
    #
    # exit()
    from dgl.nn import PGExplainer
    import dgl
    import os
    import networkx as nx
    import matplotlib.pyplot as plt

    explainer = PGExplainer(model, hidden_size, device='cpu')
    explainer.train_explanation_network(dataset)

    for idx, (graph, l) in enumerate(dataset):
        if idx == 156:
                print(f'{idx} / {len(dataset)}')
                feat = graph.ndata["attr"]
                emb = model(graph, feat, graph=False)
                probs, edge_weight = explainer.explain_graph(graph, feat, emb.data.cpu())
                print(probs)
                print(edge_weight)

                top_k = 10
                top_k_indices = np.argsort(-edge_weight)[:top_k]
                edge_mask = np.zeros(graph.num_edges(), dtype=np.float32)
                edge_mask[top_k_indices] = 1

                _, predicted = torch.max(probs, 1)
                if predicted == l:
                    print(f'Correct Prediction {l} {predicted}')

                print(edge_mask)
                graph.edata['mask'] = torch.tensor(edge_mask,
                                                   dtype=torch.float32,
                                                   device=graph.device)
                G = dgl.to_networkx(graph,
                                    node_attrs=['label'],
                                    edge_attrs=['mask'])
                plt.figure(figsize=[15, 15])

                label_dict_nodes = {
                    0: 'C',
                    1: 'N',
                    2: 'O',
                    3: 'F',
                    4: 'I',
                    5: 'Cl',
                    6: 'Br'
                }

                label_dict_edges = {
                    0: 'aromatic',
                    1: 'S',
                    2: 'D',
                    3: 'T',
                }

                # Draw the graph with edge colors based on the mask feature
                pos = nx.spring_layout(G)
                edge_colors = ['r' if mask else 'k'
                               for _, _, mask in G.edges(data='mask')]
                edge_widths = [4.0 if mask else 1.0
                               for _, _, mask in G.edges(data='mask')]
                nx.draw(G,
                        pos,
                        width=edge_widths,
                        edge_color=edge_colors,
                        with_labels=True,
                        labels={indx: label_dict_nodes[graph.ndata['label'].tolist()[indx]]
                                for indx in G.nodes()},
                        font_size=22,
                        font_color="yellow",
                        node_size=[1000 for _ in range(graph.num_nodes())],
                        )

                plt.savefig(os.path.join("mutag_img", f"{idx}_MUTAG_explain.png"), format="PNG")
                plt.show()
