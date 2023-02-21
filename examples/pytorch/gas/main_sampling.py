import argparse

import dgl
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from dataloader import GASDataset
from model_sampling import GAS
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def evaluate(model, loss_fn, dataloader, device="cpu"):
    loss = 0
    f1 = 0
    auc = 0
    rap = 0
    num_blocks = 0
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        edge_subgraph = edge_subgraph.to(device)
        u_feat = blocks[0].srcdata["feat"]["u"]
        v_feat = blocks[0].srcdata["feat"]["v"]
        f_feat = blocks[0].edges["forward"].data["feat"]
        b_feat = blocks[0].edges["backward"].data["feat"]
        labels = edge_subgraph.edges["forward"].data["label"].long()
        logits = model(edge_subgraph, blocks, f_feat, b_feat, u_feat, v_feat)

        loss += loss_fn(logits, labels).item()
        f1 += f1_score(labels.cpu(), logits.argmax(dim=1).cpu())
        auc += roc_auc_score(labels.cpu(), logits[:, 1].detach().cpu())
        pre, re, _ = precision_recall_curve(
            labels.cpu(), logits[:, 1].detach().cpu()
        )
        rap += re[pre > args.precision].max()
        num_blocks += 1

    return (
        rap / num_blocks,
        f1 / num_blocks,
        auc / num_blocks,
        loss / num_blocks,
    )


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load dataset
    dataset = GASDataset(args.dataset)
    graph = dataset[0]

    # generate mini-batch only for forward edges
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    tr_eid_dict = {}
    val_eid_dict = {}
    test_eid_dict = {}
    tr_eid_dict["forward"] = (
        graph.edges["forward"].data["train_mask"].nonzero().squeeze()
    )
    val_eid_dict["forward"] = (
        graph.edges["forward"].data["val_mask"].nonzero().squeeze()
    )
    test_eid_dict["forward"] = (
        graph.edges["forward"].data["test_mask"].nonzero().squeeze()
    )

    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    tr_loader = dgl.dataloading.DataLoader(
        graph,
        tr_eid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = dgl.dataloading.DataLoader(
        graph,
        val_eid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = dgl.dataloading.DataLoader(
        graph,
        test_eid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # binary classification
    num_classes = dataset.num_classes

    # Extract node features
    e_feats = graph.edges["forward"].data["feat"].shape[-1]
    u_feats = graph.nodes["u"].data["feat"].shape[-1]
    v_feats = graph.nodes["v"].data["feat"].shape[-1]

    # Step 2: Create model =================================================================== #
    model = GAS(
        e_in_dim=e_feats,
        u_in_dim=u_feats,
        v_in_dim=v_feats,
        e_hid_dim=args.e_hid_dim,
        u_hid_dim=args.u_hid_dim,
        v_hid_dim=args.v_hid_dim,
        out_dim=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=F.relu,
    )

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        model.train()
        tr_loss = 0
        tr_f1 = 0
        tr_auc = 0
        tr_rap = 0
        tr_blocks = 0
        for input_nodes, edge_subgraph, blocks in tr_loader:
            blocks = [b.to(device) for b in blocks]
            edge_subgraph = edge_subgraph.to(device)
            u_feat = blocks[0].srcdata["feat"]["u"]
            v_feat = blocks[0].srcdata["feat"]["v"]
            f_feat = blocks[0].edges["forward"].data["feat"]
            b_feat = blocks[0].edges["backward"].data["feat"]
            labels = edge_subgraph.edges["forward"].data["label"].long()
            logits = model(
                edge_subgraph, blocks, f_feat, b_feat, u_feat, v_feat
            )

            # compute loss
            batch_loss = loss_fn(logits, labels)
            tr_loss += batch_loss.item()
            tr_f1 += f1_score(labels.cpu(), logits.argmax(dim=1).cpu())
            tr_auc += roc_auc_score(labels.cpu(), logits[:, 1].detach().cpu())
            tr_pre, tr_re, _ = precision_recall_curve(
                labels.cpu(), logits[:, 1].detach().cpu()
            )
            tr_rap += tr_re[tr_pre > args.precision].max()
            tr_blocks += 1

            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_rap, val_f1, val_auc, val_loss = evaluate(
            model, loss_fn, val_loader, device
        )

        # Print out performance
        print(
            "In epoch {}, Train R@P: {:.4f} | Train F1: {:.4f} | Train AUC: {:.4f} | Train Loss: {:.4f}; "
            "Valid R@P: {:.4f} | Valid F1: {:.4f} | Valid AUC: {:.4f} | Valid loss: {:.4f}".format(
                epoch,
                tr_rap / tr_blocks,
                tr_f1 / tr_blocks,
                tr_auc / tr_blocks,
                tr_loss / tr_blocks,
                val_rap,
                val_f1,
                val_auc,
                val_loss,
            )
        )

    # Test with mini batch after all epoch
    model.eval()
    test_rap, test_f1, test_auc, test_loss = evaluate(
        model, loss_fn, test_loader, device
    )
    print(
        "Test R@P: {:.4f} | Test F1: {:.4f} | Test AUC: {:.4f} | Test loss: {:.4f}".format(
            test_rap, test_f1, test_auc, test_loss
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN-based Anti-Spam Model")
    parser.add_argument(
        "--dataset", type=str, default="pol", help="'pol', or 'gos'"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--e_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for edges",
    )
    parser.add_argument(
        "--u_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for source nodes",
    )
    parser.add_argument(
        "--v_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for destination nodes",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GCN layers"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=100,
        help="The max number of epochs. Default: 100",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate. Default: 1e-3"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate. Default: 0.0"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Size of mini-batches. Default: 64",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of node dataloader"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight Decay. Default: 0.0005",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.9,
        help="The value p in recall@p precision. Default: 0.9",
    )

    args = parser.parse_args()
    print(args)
    main(args)
