import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl import apply_each
from dgl.dataloading import DataLoader, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset


class HeteroGAT(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(in_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hid_size, out_size)  # Should be HeteroLinear

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # One thing is that h might return tensors with zero rows if the number of dst nodes
            # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h["paper"])


def evaluate(num_classes, model, dataloader, desc):
    preds = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm.tqdm(
            dataloader, desc=desc
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]["paper"][:, 0]
            y_hat = model(blocks, x)
            preds.append(y_hat.cpu())
            labels.append(y.cpu())
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        acc = MF.accuracy(
            preds, labels, task="multiclass", num_classes=num_classes
        )
        return acc


def train(train_loader, val_loader, test_loader, num_classes, model):
    # loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            tqdm.tqdm(train_dataloader, desc="Train")
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]["paper"][:, 0]
            y_hat = model(blocks, x)
            loss = loss_fcn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        val_acc = evaluate(num_classes, model, val_dataloader, "Val. ")
        test_acc = evaluate(num_classes, model, test_dataloader, "Test ")
        print(
            f"Epoch {epoch:05d} | Loss {total_loss/(it+1):.4f} | Validation Acc. {val_acc.item():.4f} | Test Acc. {test_acc.item():.4f}"
        )


if __name__ == "__main__":
    print(
        f"Training with DGL built-in HeteroGraphConv using GATConv as its convolution sub-modules"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load and preprocess dataset
    print("Loading data")
    dataset = DglNodePropPredDataset("ogbn-mag")
    graph, labels = dataset[0]
    graph.ndata["label"] = labels
    # add reverse edges in "cites" relation, and add reverse edge types for the rest etypes
    graph = dgl.AddReverse()(graph)
    # precompute the author, topic, and institution features
    graph.update_all(
        fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="rev_writes"
    )
    graph.update_all(
        fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="has_topic"
    )
    graph.update_all(
        fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="affiliated_with"
    )
    # find train/val/test indexes
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    train_idx = apply_each(train_idx, lambda x: x.to(device))
    val_idx = apply_each(val_idx, lambda x: x.to(device))
    test_idx = apply_each(test_idx, lambda x: x.to(device))

    # create RGAT model
    in_size = graph.ndata["feat"]["paper"].shape[1]
    num_classes = dataset.num_classes
    model = HeteroGAT(graph.etypes, in_size, 256, num_classes).to(device)

    # dataloader + model training + testing
    train_sampler = NeighborSampler(
        [5, 5, 5],
        prefetch_node_feats={k: ["feat"] for k in graph.ntypes},
        prefetch_labels={"paper": ["label"]},
    )
    val_sampler = NeighborSampler(
        [10, 10, 10],
        prefetch_node_feats={k: ["feat"] for k in graph.ntypes},
        prefetch_labels={"paper": ["label"]},
    )
    train_dataloader = DataLoader(
        graph,
        train_idx,
        train_sampler,
        device=device,
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        graph,
        val_idx,
        val_sampler,
        device=device,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        graph,
        test_idx,
        val_sampler,
        device=device,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )

    train(train_dataloader, val_dataloader, test_dataloader, num_classes, model)
