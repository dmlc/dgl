import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GraphConv


class GraphConvNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphConvNet, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.conv3 = GraphConv(hidden_feats, hidden_feats)
        self.conv4 = GraphConv(hidden_feats, hidden_feats)
        self.conv5 = GraphConv(hidden_feats, out_feats)
        self.fc = nn.Linear(out_feats, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, g, h, graph=True, edge_weight=None):
        h = self.conv1(g, h, edge_weight=edge_weight)
        h = self.conv2(g, h, edge_weight=edge_weight)
        h = self.conv3(g, h, edge_weight=edge_weight)
        h = self.conv4(g, h, edge_weight=edge_weight)
        if graph:
            h = self.conv5(g, h, edge_weight=edge_weight)
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return torch.sigmoid(self.fc(hg))
        else:
            return h


def train(train_loader, val_loader, device, model, epochs=350, lr=0.005):

    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):

            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            feat = batched_graph.ndata["attr"]
            logits = model(batched_graph, feat)

            loss = loss_fcn(logits, labels.float().unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # scheduler.step()
        with torch.no_grad():
            train_acc = evaluate(train_loader, device, model)
            valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )


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

        predicted = logits.squeeze(1) > 0.5
        total_correct += (predicted == labels).sum().item()

    acc = 1.0 * total_correct / total
    return acc


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

    model = GraphConvNet(in_size, hidden_size, out_size).to(device)

    # # model training/validating
    # print("Training...")
    # train(train_loader, val_loader, device, model)
    #
    # print("Evaluating...")
    # print(f"acc: {round(evaluate(train_loader, device, model), 2)}")
    #
    # torch.save(model, 'model.dt')

    model = torch.load('model.dt')
    model.eval()

    from dgl.nn import PGExplainer
    import dgl
    import os
    import networkx as nx
    import matplotlib.pyplot as plt

    explainer = PGExplainer(model, hidden_size, device='cpu', epochs=40)
    explainer.train_explanation_network(dataset, lambda g: g.ndata["attr"])

    for idx, (graph, l) in enumerate(dataset):
        if idx == 156:
                print(f'{idx} / {len(dataset)}')
                feat = graph.ndata["attr"]
                emb = model(graph, feat, graph=False)
                probs, edge_weight = explainer.explain_graph(graph, feat, emb.data.cpu())
                print(probs)
                print(edge_weight)

                top_k = 20
                top_k_indices = np.argsort(-edge_weight)[:top_k]
                edge_mask = np.zeros(graph.num_edges(), dtype=np.float32)
                edge_mask[top_k_indices] = 1

                predicted = probs < 0.5
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
                edge_colors = ['r' if mask else 'b'
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
