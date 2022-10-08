import argparse
import copy
import time
import traceback

import numpy as np
import torch
from data_preprocess import (
    TemporalDataset,
    TemporalRedditDataset,
    TemporalWikipediaDataset,
)
from dataloading import (
    FastTemporalEdgeCollator,
    FastTemporalSampler,
    SimpleTemporalEdgeCollator,
    SimpleTemporalSampler,
    TemporalEdgeCollator,
    TemporalEdgeDataLoader,
    TemporalSampler,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from tgn import TGN

import dgl

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)


def train(model, dataloader, sampler, criterion, optimizer, args):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        optimizer.zero_grad()
        pred_pos, pred_neg = model.embed(
            positive_pair_g, negative_pair_g, blocks
        )
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        retain_graph = True if batch_cnt == 0 and not args.fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.detach_memory()
        if not args.not_use_memory:
            model.update_memory(positive_pair_g)
        if args.fast_mode:
            sampler.attach_last_update(model.memory.last_update_t)
        print("Batch: ", batch_cnt, "Time: ", time.time() - last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss


def test_val(model, dataloader, sampler, criterion, args):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for _, postive_pair_g, negative_pair_g, blocks in dataloader:
            pred_pos, pred_neg = model.embed(
                postive_pair_g, negative_pair_g, blocks
            )
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * batch_size
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))],
                dim=0,
            )
            if not args.not_use_memory:
                model.update_memory(postive_pair_g)
            if args.fast_mode:
                sampler.attach_last_update(model.memory.last_update_t)
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="epochs for training on entire dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Size of each batch"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Embedding dim for link prediction",
    )
    parser.add_argument(
        "--memory_dim", type=int, default=100, help="dimension of memory"
    )
    parser.add_argument(
        "--temporal_dim",
        type=int,
        default=100,
        help="Temporal dimension for time encoding",
    )
    parser.add_argument(
        "--memory_updater",
        type=str,
        default="gru",
        help="Recurrent unit for memory update",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="last",
        help="Aggregation method for memory update",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="number of neighbors while doing embedding",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="topk",
        help="In embedding how node aggregate from its neighor",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of heads for multihead attention mechanism",
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        default=False,
        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained",
    )
    parser.add_argument(
        "--simple_mode",
        action="store_true",
        default=False,
        help="Simple Mode directly delete the temporal edges from the original static graph",
    )
    parser.add_argument(
        "--num_negative_samples",
        type=int,
        default=1,
        help="number of negative samplers per positive samples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikipedia",
        help="dataset selection wikipedia/reddit",
    )
    parser.add_argument(
        "--k_hop", type=int, default=1, help="sampling k-hop neighborhood"
    )
    parser.add_argument(
        "--not_use_memory",
        action="store_true",
        default=False,
        help="Enable memory for TGN Model disable memory for TGN Model",
    )

    args = parser.parse_args()

    assert not (
        args.fast_mode and args.simple_mode
    ), "you can only choose one sampling mode"
    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    if args.dataset == "wikipedia":
        data = TemporalWikipediaDataset()
    elif args.dataset == "reddit":
        data = TemporalRedditDataset()
    else:
        print("Warning Using Untested Dataset: " + args.dataset)
        data = TemporalDataset(args.dataset)

    # Pre-process data, mask new node in test set from original graph
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    num_edges = data.num_edges()
    trainval_div = int(VALID_SPLIT * num_edges)

    # Select new node from test set and remove them from entire graph
    test_split_ts = data.edata["timestamp"][trainval_div]
    test_nodes = (
        torch.cat(
            [data.edges()[0][trainval_div:], data.edges()[1][trainval_div:]]
        )
        .unique()
        .numpy()
    )
    test_new_nodes = np.random.choice(
        test_nodes, int(0.1 * len(test_nodes)), replace=False
    )

    in_subg = dgl.in_subgraph(data, test_new_nodes)
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][
        in_subg.edata["timestamp"] < test_split_ts
    ]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][
        out_subg.edata["timestamp"] < test_split_ts
    ]
    new_node_eid_delete = torch.cat(
        [new_node_in_eid_delete, new_node_out_eid_delete]
    ).unique()

    graph_new_node = copy.deepcopy(data)
    # relative order preseved
    graph_new_node.remove_edges(new_node_eid_delete)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(data)
    graph_no_new_node.remove_edges(eid_delete)

    # graph_no_new_node and graph_new_node should have same set of nid

    # Sampler Initialization
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(data, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples
    )
    # Set Train, validation, test and new node test id
    train_seed = torch.arange(int(TRAIN_SPLIT * graph_no_new_node.num_edges()))
    valid_seed = torch.arange(
        int(TRAIN_SPLIT * graph_no_new_node.num_edges()),
        trainval_div - new_node_eid_delete.size(0),
    )
    test_seed = torch.arange(
        trainval_div - new_node_eid_delete.size(0),
        graph_no_new_node.num_edges(),
    )
    test_new_node_seed = torch.arange(
        trainval_div - new_node_eid_delete.size(0), graph_new_node.num_edges()
    )

    g_sampling = (
        None
        if args.fast_mode
        else dgl.add_reverse_edges(graph_no_new_node, copy_edata=True)
    )
    new_node_g_sampling = (
        None
        if args.fast_mode
        else dgl.add_reverse_edges(graph_new_node, copy_edata=True)
    )
    if not args.fast_mode:
        new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
        g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    train_dataloader = TemporalEdgeDataLoader(
        graph_no_new_node,
        train_seed,
        sampler,
        batch_size=args.batch_size,
        negative_sampler=neg_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collator=edge_collator,
        g_sampling=g_sampling,
    )

    valid_dataloader = TemporalEdgeDataLoader(
        graph_no_new_node,
        valid_seed,
        sampler,
        batch_size=args.batch_size,
        negative_sampler=neg_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collator=edge_collator,
        g_sampling=g_sampling,
    )

    test_dataloader = TemporalEdgeDataLoader(
        graph_no_new_node,
        test_seed,
        sampler,
        batch_size=args.batch_size,
        negative_sampler=neg_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collator=edge_collator,
        g_sampling=g_sampling,
    )

    test_new_node_dataloader = TemporalEdgeDataLoader(
        graph_new_node,
        test_new_node_seed,
        new_node_sampler if args.fast_mode else sampler,
        batch_size=args.batch_size,
        negative_sampler=neg_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collator=edge_collator,
        g_sampling=new_node_g_sampling,
    )

    edge_dim = data.edata["feats"].shape[1]
    num_node = data.num_nodes()

    model = TGN(
        edge_feat_dim=edge_dim,
        memory_dim=args.memory_dim,
        temporal_dim=args.temporal_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_nodes=num_node,
        n_neighbors=args.n_neighbors,
        memory_updater_type=args.memory_updater,
        layers=args.k_hop,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Implement Logging mechanism
    f = open("logging.txt", "w")
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_loss = train(
                model, train_dataloader, sampler, criterion, optimizer, args
            )

            val_ap, val_auc = test_val(
                model, valid_dataloader, sampler, criterion, args
            )
            memory_checkpoint = model.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            test_ap, test_auc = test_val(
                model, test_dataloader, sampler, criterion, args
            )
            model.restore_memory(memory_checkpoint)
            if args.fast_mode:
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap, nn_test_auc = test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args
            )
            log_content = []
            log_content.append(
                "Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                    i, train_loss, val_ap, val_auc
                )
            )
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(
                    i, test_ap, test_auc
                )
            )
            log_content.append(
                "Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                    i, nn_test_ap, nn_test_auc
                )
            )

            f.writelines(log_content)
            model.reset_memory()
            if i < args.epochs - 1 and args.fast_mode:
                sampler.reset()
            print(log_content[0], log_content[1], log_content[2])
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
