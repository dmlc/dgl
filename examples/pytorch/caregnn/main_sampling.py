import argparse

import dgl

import torch as th
import torch.optim as optim
from model_sampling import _l1_dist, CAREGNN, CARESampler
from sklearn.metrics import recall_score, roc_auc_score
from torch.nn.functional import softmax
from utils import EarlyStopping


def evaluate(model, loss_fn, dataloader, device="cpu"):
    loss = 0
    auc = 0
    recall = 0
    num_blocks = 0
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        feature = blocks[0].srcdata["feature"]
        label = blocks[-1].dstdata["label"]
        logits_gnn, logits_sim = model(blocks, feature)

        # compute loss
        loss += (
            loss_fn(logits_gnn, label).item()
            + args.sim_weight * loss_fn(logits_sim, label).item()
        )
        recall += recall_score(
            label.cpu(), logits_gnn.argmax(dim=1).detach().cpu()
        )
        auc += roc_auc_score(
            label.cpu(), softmax(logits_gnn, dim=1)[:, 1].detach().cpu()
        )
        num_blocks += 1

    return recall / num_blocks, auc / num_blocks, loss / num_blocks


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load dataset
    dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4)
    graph = dataset[0]
    num_classes = dataset.num_classes

    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
        args.num_workers = 0
    else:
        device = "cpu"

    # retrieve labels of ground truth
    labels = graph.ndata["label"].to(device)

    # Extract node features
    feat = graph.ndata["feature"].to(device)
    layers_feat = feat.expand(args.num_layers, -1, -1)

    # retrieve masks for train/validation/test
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    # Reinforcement learning module only for positive training nodes
    rl_idx = th.nonzero(
        train_mask.to(device) & labels.bool(), as_tuple=False
    ).squeeze(1)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = CAREGNN(
        in_dim=feat.shape[-1],
        num_classes=num_classes,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        activation=th.tanh,
        step_size=args.step_size,
        edges=graph.canonical_etypes,
    )

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    _, cnt = th.unique(labels, return_counts=True)
    loss_fn = th.nn.CrossEntropyLoss(weight=1 / cnt)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        # calculate the distance of each edges and sample based on the distance
        dists = []
        p = []
        for i in range(args.num_layers):
            dist = {}
            graph.ndata["nd"] = th.tanh(model.layers[i].MLP(layers_feat[i]))
            for etype in graph.canonical_etypes:
                graph.apply_edges(_l1_dist, etype=etype)
                dist[etype] = graph.edges[etype].data.pop("ed").detach().cpu()
            dists.append(dist)
            p.append(model.layers[i].p)
        graph.ndata.pop("nd")
        sampler = CARESampler(p, dists, args.num_layers)

        # train
        model.train()
        tr_loss = 0
        tr_recall = 0
        tr_auc = 0
        tr_blk = 0
        train_dataloader = dgl.dataloading.DataLoader(
            graph,
            train_idx,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
        )

        for input_nodes, output_nodes, blocks in train_dataloader:
            blocks = [b.to(device) for b in blocks]
            train_feature = blocks[0].srcdata["feature"]
            train_label = blocks[-1].dstdata["label"]
            logits_gnn, logits_sim = model(blocks, train_feature)

            # compute loss
            blk_loss = loss_fn(
                logits_gnn, train_label
            ) + args.sim_weight * loss_fn(logits_sim, train_label)
            tr_loss += blk_loss.item()
            tr_recall += recall_score(
                train_label.cpu(), logits_gnn.argmax(dim=1).detach().cpu()
            )
            tr_auc += roc_auc_score(
                train_label.cpu(),
                softmax(logits_gnn, dim=1)[:, 1].detach().cpu(),
            )
            tr_blk += 1

            # backward
            optimizer.zero_grad()
            blk_loss.backward()
            optimizer.step()

        # Reinforcement learning module
        model.RLModule(graph, epoch, rl_idx, dists)

        # validation
        model.eval()
        val_dataloader = dgl.dataloading.DataLoader(
            graph,
            val_idx,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
        )

        val_recall, val_auc, val_loss = evaluate(
            model, loss_fn, val_dataloader, device
        )

        # Print out performance
        print(
            "In epoch {}, Train Recall: {:.4f} | Train AUC: {:.4f} | Train Loss: {:.4f}; "
            "Valid Recall: {:.4f} | Valid AUC: {:.4f} | Valid loss: {:.4f}".format(
                epoch,
                tr_recall / tr_blk,
                tr_auc / tr_blk,
                tr_loss / tr_blk,
                val_recall,
                val_auc,
                val_loss,
            )
        )

        if args.early_stop:
            if stopper.step(val_auc, model):
                break

    # Test with mini batch after all epoch
    model.eval()
    if args.early_stop:
        model.load_state_dict(th.load("es_checkpoint.pt"))
    test_dataloader = dgl.dataloading.DataLoader(
        graph,
        test_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    test_recall, test_auc, test_loss = evaluate(
        model, loss_fn, test_dataloader, device
    )

    print(
        "Test Recall: {:.4f} | Test AUC: {:.4f} | Test loss: {:.4f}".format(
            test_recall, test_auc, test_loss
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN-based Anti-Spam Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon",
        help="DGL dataset for this model (yelp, or amazon)",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=64, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Size of mini-batch"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=30,
        help="The max number of epochs. Default: 30",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate. Default: 0.01"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay. Default: 0.001",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.02,
        help="RL action step size (lambda 2). Default: 0.02",
    )
    parser.add_argument(
        "--sim_weight",
        type=float,
        default=2,
        help="Similarity loss weight (lambda 1). Default: 0.001",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of node dataloader"
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop",
    )

    args = parser.parse_args()
    th.manual_seed(717)
    print(args)
    main(args)
