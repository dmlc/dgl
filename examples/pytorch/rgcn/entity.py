import argparse

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from dgl.nn.pytorch import RelGraphConv
from torchmetrics.functional import accuracy


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )

    def forward(self, g):
        x = self.emb.weight
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"]))
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        return h


def evaluate(g, target_idx, labels, num_classes, test_mask, model):
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    model.eval()
    with torch.no_grad():
        logits = model(g)
    logits = logits[target_idx]
    return accuracy(
        logits[test_idx].argmax(dim=1),
        labels[test_idx],
        task="multiclass",
        num_classes=num_classes,
    ).item()


def train(g, target_idx, labels, num_classes, train_mask, model):
    # define train idx, loss function and optimizer
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    model.train()
    for epoch in range(50):
        logits = model(g)
        logits = logits[target_idx]
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(
            logits[train_idx].argmax(dim=1),
            labels[train_idx],
            task="multiclass",
            num_classes=num_classes,
        ).item()
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RGCN for entity classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aifb",
        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with DGL built-in RGCN module.")

    # load and preprocess dataset
    if args.dataset == "aifb":
        data = AIFBDataset()
    elif args.dataset == "mutag":
        data = MUTAGDataset()
    elif args.dataset == "bgs":
        data = BGSDataset()
    elif args.dataset == "am":
        data = AMDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    g = g.int().to(device)
    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    labels = g.nodes[category].data.pop("labels")
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    # calculate normalization weight for each edge, and find target category and node id
    for cetype in g.canonical_etypes:
        g.edges[cetype].data["norm"] = dgl.norm_by_dst(g, cetype).unsqueeze(1)
    category_id = g.ntypes.index(category)
    g = dgl.to_homogeneous(g, edata=["norm"])
    node_ids = torch.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    # create RGCN model
    in_size = g.num_nodes()  # featureless with one-hot encoding
    num_classes = data.num_classes
    model = RGCN(in_size, 16, num_classes, num_rels).to(device)

    train(g, target_idx, labels, num_classes, train_mask, model)
    acc = evaluate(g, target_idx, labels, num_classes, test_mask, model)
    print("Test accuracy {:.4f}".format(acc))
