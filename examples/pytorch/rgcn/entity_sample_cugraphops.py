import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
from dgl.contrib.cugraph.nn import RelGraphConv
import argparse


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, fanouts):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            fanouts[0],
            regularizer="basis",
            num_bases=num_bases,
            self_loop=False,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            fanouts[1],
            regularizer="basis",
            num_bases=num_bases,
            self_loop=False,
        )

    def forward(self, g):
        x = self.emb(g[0].srcdata[dgl.NID])
        h = F.relu(self.conv1(g[0], x, g[0].edata[dgl.ETYPE], norm=g[0].edata["norm"]))
        h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], norm=g[1].edata["norm"])
        return h

    def update_fanouts(self, fanouts):
        self.conv1.fanout = fanouts[0]
        self.conv2.fanout = fanouts[1]


def evaluate(model, labels, dataloader, inv_target):
    model.eval()
    eval_logits = []
    eval_seeds = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            output_nodes = inv_target[output_nodes.type(torch.int64)]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(output_nodes.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    return accuracy(eval_logits.argmax(dim=1), labels[eval_seeds].cpu()).item()


def train(device, g, target_idx, labels, train_mask, model, fanouts):
    # define train idx, loss function and optimizer
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # construct sampler and dataloader
    sampler = MultiLayerNeighborSampler(fanouts)
    train_loader = DataLoader(
        g,
        target_idx[train_idx].type(g.idtype),
        sampler,
        device=device,
        batch_size=100,
        shuffle=True,
    )
    # no separate validation subset, use train index instead for validation
    val_loader = DataLoader(
        g,
        target_idx[train_idx].type(g.idtype),
        sampler,
        device=device,
        batch_size=100,
        shuffle=False,
    )
    model.train()
    for epoch in range(50):
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            output_nodes = inv_target[output_nodes.type(torch.int64)]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            loss = loss_fcn(logits, labels[output_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate(model, labels, val_loader, inv_target)
        print(
            "Epoch {:05d} | Loss {:.4f} | Val. Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RGCN for entity classification with sampling"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aifb",
        help="Dataset name ('aifb', 'mutag', 'bgs', 'am'), default to 'aifb'.",
    )
    parser.add_argument(
        "--idtype",
        type=str,
        default="int64",
        help="Graph idtype ('int64' or 'int32'), default to 'int64'.",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    print(f"Training with DGL cugraph.nn RGCN module with sampling.")

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
    hg = data[0].to(device)
    hg = hg.int() if args.idtype == "int32" else hg.long()
    num_rels = len(hg.canonical_etypes)
    category = data.predict_category
    labels = hg.nodes[category].data.pop("labels")
    train_mask = hg.nodes[category].data.pop("train_mask")
    test_mask = hg.nodes[category].data.pop("test_mask")
    # find target category and node id
    category_id = hg.ntypes.index(category)
    g = dgl.to_homogeneous(hg)
    node_ids = torch.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    # rename the field as they can be changed by DataLoader
    g.ndata["ntype"] = g.ndata.pop(dgl.NTYPE)
    g.ndata["type_id"] = g.ndata.pop(dgl.NID)
    # find the mapping (inv_target) from global node IDs to type-specific node IDs
    inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64).to(device)
    inv_target[target_idx] = torch.arange(
        0, target_idx.shape[0], dtype=inv_target.dtype
    ).to(device)

    # create RGCN model
    in_size = g.num_nodes()  # featureless with one-hot encoding
    out_size = data.num_classes
    num_bases = 20
    fanouts = [4, 4]
    model = RGCN(in_size, 16, out_size, num_rels, num_bases, fanouts).to(device)

    train(device, g, target_idx, labels, train_mask, model, fanouts)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    # Note: cugraph-ops aggregators are designed for sampled graphs (MFGs) and expect fanout
    # as input for performance considerations. Hence, we have to update the fanouts during evaluation.
    # Setting fanouts to -1 for sampling all neighbors is not supported.
    test_sampler = MultiLayerNeighborSampler([500, 500])
    model.update_fanouts(test_sampler.fanouts)
    test_loader = DataLoader(
        g,
        target_idx[test_idx].type(g.idtype),
        test_sampler,
        device=device,
        batch_size=32,
        shuffle=False,
    )
    acc = evaluate(model, labels, test_loader, inv_target)
    print("Test accuracy {:.4f}".format(acc))
