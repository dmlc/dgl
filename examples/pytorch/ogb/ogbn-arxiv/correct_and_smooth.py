import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F

from dgl import function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.num_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.num_edges()}")

    graph.create_formats_()

    return graph


def general_outcome_correlation(
    graph, y0, n_prop=50, alpha=0.8, use_norm=False, post_step=None
):
    with graph.local_scope():
        y = y0
        for _ in range(n_prop):
            if use_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (y.dim() - 1)
                norm = torch.reshape(norm, shp)
                y = y * norm

            graph.srcdata.update({"y": y})
            graph.update_all(fn.copy_u("y", "m"), fn.mean("m", "y"))
            y = graph.dstdata["y"]

            if use_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (y.dim() - 1)
                norm = torch.reshape(norm, shp)
                y = y * norm

            y = alpha * y + (1 - alpha) * y0

            if post_step is not None:
                y = post_step(y)

        return y


def evaluate(labels, pred, train_idx, val_idx, test_idx, evaluator):
    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
    )


def run(args, graph, labels, pred, train_idx, val_idx, test_idx, evaluator):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    y = pred.clone()
    y[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1)
    # dy = torch.zeros(graph.num_nodes(), n_classes, device=device)
    # dy[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1) - pred[train_idx]

    _train_acc, val_acc, test_acc = evaluate(
        labels, y, train_idx, val_idx, test_idx, evaluator_wrapper
    )

    # print("train acc:", _train_acc)
    print("original val acc:", val_acc)
    print("original test acc:", test_acc)

    # NOTE: Only "smooth" is performed here.
    # smoothed_dy = general_outcome_correlation(
    #     graph, dy, alpha=args.alpha1, use_norm=args.use_norm, post_step=lambda x: x.clamp(-1, 1)
    # )

    # y[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1)
    # smoothed_dy = smoothed_dy
    # y = y + args.alpha2 * smoothed_dy  # .clamp(0, 1)

    smoothed_y = general_outcome_correlation(
        graph,
        y,
        alpha=args.alpha,
        use_norm=args.use_norm,
        post_step=lambda x: x.clamp(0, 1),
    )

    _train_acc, val_acc, test_acc = evaluate(
        labels, smoothed_y, train_idx, val_idx, test_idx, evaluator_wrapper
    )

    # print("train acc:", _train_acc)
    print("val acc:", val_acc)
    print("test acc:", test_acc)

    return val_acc, test_acc


def main():
    global device

    argparser = argparse.ArgumentParser(description="implementation of C&S)")
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides --gpu.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--use-norm",
        action="store_true",
        help="Use symmetrically normalized adjacency matrix.",
    )
    argparser.add_argument("--alpha", type=float, default=0.6, help="alpha")
    argparser.add_argument(
        "--pred-files",
        type=str,
        default="./output/*.pt",
        help="address of prediction files",
    )
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    # run
    val_accs, test_accs = [], []

    for pred_file in glob.iglob(args.pred_files):
        print("load:", pred_file)
        pred = torch.load(pred_file)
        val_acc, test_acc = run(
            args, graph, labels, pred, train_idx, val_idx, test_idx, evaluator
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(args)
    print(f"Runned {len(val_accs)} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")


if __name__ == "__main__":
    main()

# Namespace(alpha=0.6, cpu=False, gpu=0, pred_files='./output/*.pt', use_norm=True)
# Runned 20 times
# Val Accs: [0.7523742407463337, 0.750729890264774, 0.7524077989194268, 0.7527098224772644, 0.752508473438706, 0.7509983556495184, 0.751904426323031, 0.7514010537266351, 0.7524077989194268, 0.753716567670056, 0.7523071244001477, 0.7518373099768448, 0.7528440551696366, 0.7509983556495184, 0.7521057753615893, 0.7520386590154032, 0.7500251686298198, 0.7513674955535421, 0.7509312393033323, 0.7518037518037518]
# Test Accs: [0.7392753533732486, 0.7381437359833755, 0.7412093903668497, 0.7402629467316832, 0.7386169578009588, 0.7380408616752052, 0.7397280003291978, 0.7401189227002448, 0.7424233072032591, 0.7397280003291978, 0.7378351130588647, 0.7400160483920746, 0.740921342303973, 0.7385758080776906, 0.7411682406435817, 0.7389667304487377, 0.7396457008826616, 0.7384935086311545, 0.7396251260210275, 0.7379997119519371]
# Average val accuracy: 0.751870868149938 ± 0.0008415008835817228
# Average test accuracy: 0.7395397403452462 ± 0.0012162384423867229
