import argparse
import warnings

import dgl

import torch as th
from dataset import load
from dgl.dataloading import GraphDataLoader

warnings.filterwarnings("ignore")

from model import MVGRL
from utils import linearsvc

parser = argparse.ArgumentParser(description="mvgrl")

parser.add_argument(
    "--dataname", type=str, default="MUTAG", help="Name of dataset."
)
parser.add_argument(
    "--gpu", type=int, default=-1, help="GPU index. Default: -1, using cpu."
)
parser.add_argument(
    "--epochs", type=int, default=200, help=" Number of training periods."
)
parser.add_argument(
    "--patience", type=int, default=20, help="Early stopping steps."
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="Learning rate of mvgrl."
)
parser.add_argument(
    "--wd", type=float, default=0.0, help="Weight decay of mvgrl."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument(
    "--n_layers", type=int, default=4, help="Number of GNN layers."
)
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dim.")

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"


def collate(samples):
    """collate function for building the graph dataloader"""
    graphs, diff_graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    batched_diff_graph = dgl.batch(diff_graphs)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata["graph_id"] = graph_id

    return batched_graph, batched_diff_graph, batched_labels


if __name__ == "__main__":
    # Step 1: Prepare data =================================================================== #
    dataset = load(args.dataname)

    graphs, diff_graphs, labels = map(list, zip(*dataset))
    print("Number of graphs:", len(graphs))
    # generate a full-graph with all examples for evaluation

    wholegraph = dgl.batch(graphs)
    whole_dg = dgl.batch(diff_graphs)

    # create dataloader for batch training
    dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        drop_last=False,
        shuffle=True,
    )

    in_dim = wholegraph.ndata["feat"].shape[1]

    # Step 2: Create model =================================================================== #
    model = MVGRL(in_dim, args.hid_dim, args.n_layers)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print("===== Before training ======")

    wholegraph = wholegraph.to(args.device)
    whole_dg = whole_dg.to(args.device)
    wholefeat = wholegraph.ndata.pop("feat")
    whole_weight = whole_dg.edata.pop("edge_weight")

    embs = model.get_embedding(wholegraph, whole_dg, wholefeat, whole_weight)
    lbls = th.LongTensor(labels)
    acc_mean, acc_std = linearsvc(embs, lbls)
    print("accuracy_mean, {:.4f}".format(acc_mean))

    best = float("inf")
    cnt_wait = 0
    # Step 4: Training epochs =============================================================== #
    for epoch in range(args.epochs):
        loss_all = 0
        model.train()

        for graph, diff_graph, label in dataloader:
            graph = graph.to(args.device)
            diff_graph = diff_graph.to(args.device)

            feat = graph.ndata["feat"]
            graph_id = graph.ndata["graph_id"]
            edge_weight = diff_graph.edata["edge_weight"]
            n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, diff_graph, feat, edge_weight, graph_id)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, Loss {:.4f}".format(epoch, loss_all))

        if loss_all < best:
            best = loss_all
            best_t = epoch
            cnt_wait = 0
            th.save(model.state_dict(), f"{args.dataname}.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping")
            break

    print("Training End")

    # Step 5:  Linear evaluation ========================================================== #
    model.load_state_dict(th.load(f"{args.dataname}.pkl"))
    embs = model.get_embedding(wholegraph, whole_dg, wholefeat, whole_weight)

    acc_mean, acc_std = linearsvc(embs, lbls)
    print("accuracy_mean, {:.4f}".format(acc_mean))
