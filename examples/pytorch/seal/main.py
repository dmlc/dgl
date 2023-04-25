import time

import numpy as np
import torch
import torch.multiprocessing

from dgl import EID, NID
from dgl.dataloading import GraphDataLoader
from logger import LightLogging
from model import DGCNN, GCN
from sampler import SEALData
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from utils import evaluate_hits, load_ogb_dataset, parse_arguments

torch.multiprocessing.set_sharing_strategy("file_system")

"""
Part of the code are adapted from
https://github.com/facebookresearch/SEAL_OGB
"""


def train(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    num_graphs=32,
    total_graphs=None,
):
    model.train()

    total_loss = 0
    for g, labels in tqdm(dataloader, ncols=100):
        g = g.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(g, g.ndata["z"], g.ndata[NID], g.edata[EID])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    y_pred, y_true = [], []
    for g, labels in tqdm(dataloader, ncols=100):
        g = g.to(device)
        logits = model(g, g.ndata["z"], g.ndata[NID], g.edata[EID])
        y_pred.append(logits.view(-1).cpu())
        y_true.append(labels.view(-1).cpu().to(torch.float))

    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = y_pred[y_true == 1]
    neg_pred = y_pred[y_true == 0]

    return pos_pred, neg_pred


def main(args, print_fn=print):
    print_fn("Experiment arguments: {}".format(args))

    if args.random_seed:
        torch.manual_seed(args.random_seed)
    else:
        torch.manual_seed(123)
    # Load dataset
    if args.dataset.startswith("ogbl"):
        graph, split_edge = load_ogb_dataset(args.dataset)
    else:
        raise NotImplementedError

    num_nodes = graph.num_nodes()

    # set gpu
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu_id)
    else:
        device = "cpu"

    if args.dataset == "ogbl-collab":
        # ogbl-collab dataset is multi-edge graph
        use_coalesce = True
    else:
        use_coalesce = False

    # Generate positive and negative edges and corresponding labels
    # Sampling subgraphs and generate node labeling features
    seal_data = SEALData(
        g=graph,
        split_edge=split_edge,
        hop=args.hop,
        neg_samples=args.neg_samples,
        subsample_ratio=args.subsample_ratio,
        use_coalesce=use_coalesce,
        prefix=args.dataset,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        print_fn=print_fn,
    )
    node_attribute = seal_data.ndata["feat"]
    edge_weight = seal_data.edata["weight"].float()

    train_data = seal_data("train")
    val_data = seal_data("valid")
    test_data = seal_data("test")

    train_graphs = len(train_data.graph_list)

    # Set data loader

    train_loader = GraphDataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = GraphDataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = GraphDataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # set model
    if args.model == "gcn":
        model = GCN(
            num_layers=args.num_layers,
            hidden_units=args.hidden_units,
            gcn_type=args.gcn_type,
            pooling_type=args.pooling,
            node_attributes=node_attribute,
            edge_weights=edge_weight,
            node_embedding=None,
            use_embedding=True,
            num_nodes=num_nodes,
            dropout=args.dropout,
        )
    elif args.model == "dgcnn":
        model = DGCNN(
            num_layers=args.num_layers,
            hidden_units=args.hidden_units,
            k=args.sort_k,
            gcn_type=args.gcn_type,
            node_attributes=node_attribute,
            edge_weights=edge_weight,
            node_embedding=None,
            use_embedding=True,
            num_nodes=num_nodes,
            dropout=args.dropout,
        )
    else:
        raise ValueError("Model error")

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    print_fn(
        "Total parameters: {}".format(
            sum([p.numel() for p in model.parameters()])
        )
    )

    # train and evaluate loop
    summary_val = []
    summary_test = []
    for epoch in range(args.epochs):
        start_time = time.time()
        loss = train(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_graphs=args.batch_size,
            total_graphs=train_graphs,
        )
        train_time = time.time()
        if epoch % args.eval_steps == 0:
            val_pos_pred, val_neg_pred = evaluate(
                model=model, dataloader=val_loader, device=device
            )
            test_pos_pred, test_neg_pred = evaluate(
                model=model, dataloader=test_loader, device=device
            )

            val_metric = evaluate_hits(
                args.dataset, val_pos_pred, val_neg_pred, args.hits_k
            )
            test_metric = evaluate_hits(
                args.dataset, test_pos_pred, test_neg_pred, args.hits_k
            )
            evaluate_time = time.time()
            print_fn(
                "Epoch-{}, train loss: {:.4f}, hits@{}: val-{:.4f}, test-{:.4f}, "
                "cost time: train-{:.1f}s, total-{:.1f}s".format(
                    epoch,
                    loss,
                    args.hits_k,
                    val_metric,
                    test_metric,
                    train_time - start_time,
                    evaluate_time - start_time,
                )
            )
            summary_val.append(val_metric)
            summary_test.append(test_metric)

    summary_test = np.array(summary_test)

    print_fn("Experiment Results:")
    print_fn(
        "Best hits@{}: {:.4f}, epoch: {}".format(
            args.hits_k, np.max(summary_test), np.argmax(summary_test)
        )
    )


if __name__ == "__main__":
    args = parse_arguments()
    logger = LightLogging(log_name="SEAL", log_path="./logs")
    main(args, logger.info)
