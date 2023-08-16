"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset

import dgl
import dgl.sparse as dglsp

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(
        self, ntype_num, embed_size, embed_name="embed", activation=None, dropout=0.0
    ):
        super(RelGraphEmbed, self).__init__()
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Create weight embeddings for each node for each relation.
        self.embeds = nn.ParameterDict()
        for ntype, num_nodes in ntype_num.items():
            embed = nn.Parameter(th.Tensor(num_nodes, self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self):
        return self.embeds


class HeteroRelationalGraphConv(nn.Module):
    r"""Relational graph convolution layer. 

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    relation_names : list[str]
        Relation names.
    """

    def __init__(
        self,
        in_size,
        out_size,
        relation_names,
    ):
        super(HeteroRelationalGraphConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names

        self.conv = nn.ModuleDict(
            {
                str(rel): nn.Linear(in_size, out_size)
                for rel in relation_names
            }
        )

        self.dropout = nn.Dropout(0.0)

    def forward(self, A, inputs):
        """Forward computation

        Parameters
        ----------
        A : Hetero Sparse Matrix 
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        hs = {}
        for rel in A:
            src_type, edge_type, dst_type = rel
            if dst_type not in hs:
                hs[dst_type] = th.zeros(
                    inputs[dst_type].shape[0], self.out_size)
            hs[dst_type] = hs[dst_type] + \
                (A[rel].T @ self.conv[str(edge_type)](inputs[src_type]))
            hs[dst_type] = F.relu(hs[dst_type])

        return hs


class EntityClassify(nn.Module):
    def __init__(
        self,
        in_size,
        out_dim,
        ntype_num,
        relation_names,
    ):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.out_dim = out_dim
        self.relation_names = relation_names
        self.relation_names.sort()

        self.embed_layer = RelGraphEmbed(ntype_num, self.in_size)
        self.layers = nn.ModuleList()
        # Input to hidden.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.in_size,
                self.relation_names,
            )
        )
        # Hidden to output.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.out_dim,
                self.relation_names,
            )
        )

    def forward(self, A):
        h = self.embed_layer()
        for layer in self.layers:
            h = layer(A, h)
        return h

    def inference(self, g, batch_size, device, num_workers, x=None):
        """Minibatch inference of final representation over all node types.

        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.
        """

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: th.zeros(
                    g.num_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim,
                )
                for k in g.ntypes
            }

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                {k: th.arange(g.num_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )

            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0].to(device)

                    h = {
                        k: x[k][input_nodes[k]].to(device)
                        for k in input_nodes.keys()
                    }
                    h = layer(block, h)

                    for k in output_nodes.keys():
                        y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y


def main(args):
    # Load graph data.
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")

    # Split dataset into train, validate, test.
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # Check cuda.
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # Create model.
    model = EntityClassify(
        args.n_hidden,
        num_classes,
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        list(set(g.etypes)),
    )

    if use_cuda:
        model.cuda()

    # Optimizer.
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm
    )

    # Construct sparse metrix.
    g = g.local_var()
    A = {}
    for stype, etype, dtype in g.canonical_etypes:
        eg = g[stype, etype, dtype]
        indices = th.stack(eg.edges('uv'))
        A[(stype, etype, dtype)] = dglsp.spmatrix(
            indices, shape=(g.num_nodes(stype), g.num_nodes(dtype)))

    # Training loop.
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(A)[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(
            logits[val_idx].argmax(dim=1) == labels[val_idx]
        ).item() / len(val_idx)
        print(
            f"Epoch {epoch:05d} | Train Acc: {train_acc:.4f} | "
            f"Train Loss: {loss.item():.4f} | Valid Acc: {val_acc:.4f} | "
            f"Valid loss: {val_loss.item():.4f} | Time: {np.average(dur):.4f}"
        )
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward(A)[category]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    print(
        "Test Acc: {:.4f} | Test loss: {:.4f}".format(
            test_acc, test_loss.item()
        )
    )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden units"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="path for save the model"
    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")

    # Only one of modes in this group is valid.
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
