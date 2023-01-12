import datetime
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm
from utils import *

import dgl
import dgl.function as fn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_graph(network_data, vocab):
    """Build graph, treat all nodes as the same type

    Parameters
    ----------
    network_data: a dict
        keys describing the edge types, values representing edges
    vocab: a dict
        mapping node IDs to node indices
    Output
    ------
    DGLGraph
        a heterogenous graph, with one node type and different edge types
    """
    graphs = []

    node_type = "_N"  # '_N' can be replaced by an arbitrary name
    data_dict = dict()
    num_nodes_dict = {node_type: len(vocab)}

    for edge_type in network_data:
        tmp_data = network_data[edge_type]
        src = []
        dst = []
        for edge in tmp_data:
            src.extend([vocab[edge[0]], vocab[edge[1]]])
            dst.extend([vocab[edge[1]], vocab[edge[0]]])
        data_dict[(node_type, edge_type, node_type)] = (src, dst)
    graph = dgl.heterograph(data_dict, num_nodes_dict)

    return graph


class NeighborSampler(object):
    def __init__(self, g, num_fanouts):
        self.g = g
        self.num_fanouts = num_fanouts

    def sample(self, pairs):
        pairs = np.stack(pairs)
        heads, tails, types = pairs[:, 0], pairs[:, 1], pairs[:, 2]
        seeds, head_invmap = torch.unique(
            torch.LongTensor(heads), return_inverse=True
        )
        blocks = []
        for fanout in reversed(self.num_fanouts):
            sampled_graph = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            sampled_block = dgl.to_block(sampled_graph, seeds)
            seeds = sampled_block.srcdata[dgl.NID]
            blocks.insert(0, sampled_block)
        return (
            blocks,
            torch.LongTensor(head_invmap),
            torch.LongTensor(tails),
            torch.LongTensor(types),
        )


class DGLGATNE(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_size,
        embedding_u_size,
        edge_types,
        edge_type_count,
        dim_a,
    ):
        super(DGLGATNE, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_types = edge_types
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a

        self.node_embeddings = nn.Embedding(
            num_nodes, embedding_size, sparse=True
        )
        self.node_type_embeddings = nn.Embedding(
            num_nodes * edge_type_count, embedding_u_size, sparse=True
        )
        self.trans_weights = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
        self.trans_weights_s2 = Parameter(
            torch.FloatTensor(edge_type_count, dim_a, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.weight.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.weight.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(
            std=1.0 / math.sqrt(self.embedding_size)
        )
        self.trans_weights_s1.data.normal_(
            std=1.0 / math.sqrt(self.embedding_size)
        )
        self.trans_weights_s2.data.normal_(
            std=1.0 / math.sqrt(self.embedding_size)
        )

    # embs: [batch_size, embedding_size]
    def forward(self, block):
        input_nodes = block.srcdata[dgl.NID]
        output_nodes = block.dstdata[dgl.NID]
        batch_size = block.number_of_dst_nodes()
        node_type_embed = []

        with block.local_scope():
            for i in range(self.edge_type_count):
                edge_type = self.edge_types[i]
                block.srcdata[edge_type] = self.node_type_embeddings(
                    input_nodes * self.edge_type_count + i
                )
                block.dstdata[edge_type] = self.node_type_embeddings(
                    output_nodes * self.edge_type_count + i
                )
                block.update_all(
                    fn.copy_u(edge_type, "m"),
                    fn.sum("m", edge_type),
                    etype=edge_type,
                )
                node_type_embed.append(block.dstdata[edge_type])

            node_type_embed = torch.stack(node_type_embed, 1)
            tmp_node_type_embed = node_type_embed.unsqueeze(2).view(
                -1, 1, self.embedding_u_size
            )
            trans_w = (
                self.trans_weights.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.embedding_u_size, self.embedding_size)
            )
            trans_w_s1 = (
                self.trans_weights_s1.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.embedding_u_size, self.dim_a)
            )
            trans_w_s2 = (
                self.trans_weights_s2.unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(-1, self.dim_a, 1)
            )

            attention = (
                F.softmax(
                    torch.matmul(
                        torch.tanh(
                            torch.matmul(tmp_node_type_embed, trans_w_s1)
                        ),
                        trans_w_s2,
                    )
                    .squeeze(2)
                    .view(-1, self.edge_type_count),
                    dim=1,
                )
                .unsqueeze(1)
                .repeat(1, self.edge_type_count, 1)
            )

            node_type_embed = torch.matmul(attention, node_type_embed).view(
                -1, 1, self.embedding_u_size
            )
            node_embed = self.node_embeddings(output_nodes).unsqueeze(1).repeat(
                1, self.edge_type_count, 1
            ) + torch.matmul(node_type_embed, trans_w).view(
                -1, self.edge_type_count, self.embedding_size
            )
            last_node_embed = F.normalize(node_embed, dim=2)

            return (
                last_node_embed  # [batch_size, edge_type_count, embedding_size]
            )


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size

        # [ (log(i+2) - log(i+1)) / log(num_nodes + 1)]
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1))
                    / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )
        self.weights = nn.Embedding(num_nodes, embedding_size, sparse=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.weight.data.normal_(
            std=1.0 / math.sqrt(self.embedding_size)
        )

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights(label)), 1))
        )
        negs = (
            torch.multinomial(
                self.sample_weights, self.num_sampled * n, replacement=True
            )
            .view(n, self.num_sampled)
            .to(input.device)
        )
        noise = torch.neg(self.weights(negs))
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


def run(proc_id, n_gpus, args, devices, data):
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345"
        )
        world_size = n_gpus
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            world_size=world_size,
            rank=proc_id,
            timeout=datetime.timedelta(seconds=100),
        )
    torch.cuda.set_device(dev_id)

    g, train_pairs, index2word, edge_types, num_nodes, edge_type_count = data

    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples
    num_workers = args.workers

    neighbor_sampler = NeighborSampler(g, [neighbor_samples])
    if n_gpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_pairs,
            num_replicas=world_size,
            rank=proc_id,
            shuffle=True,
            drop_last=False,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_pairs,
            batch_size=batch_size,
            collate_fn=neighbor_sampler.sample,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_pairs,
            batch_size=batch_size,
            collate_fn=neighbor_sampler.sample,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )

    model = DGLGATNE(
        num_nodes,
        embedding_size,
        embedding_u_size,
        edge_types,
        edge_type_count,
        dim_a,
    )

    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id
        )

    nsloss.to(dev_id)

    if n_gpus > 1:
        mmodel = model.module
    else:
        mmodel = model

    embeddings_params = list(
        map(id, mmodel.node_embeddings.parameters())
    ) + list(map(id, mmodel.node_type_embeddings.parameters()))
    weights_params = list(map(id, nsloss.weights.parameters()))

    optimizer = torch.optim.Adam(
        [
            {
                "params": filter(
                    lambda p: id(p) not in embeddings_params,
                    model.parameters(),
                )
            },
            {
                "params": filter(
                    lambda p: id(p) not in weights_params,
                    nsloss.parameters(),
                )
            },
        ],
        lr=2e-3,
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [
            {"params": mmodel.node_embeddings.parameters()},
            {"params": mmodel.node_type_embeddings.parameters()},
            {"params": nsloss.weights.parameters()},
        ],
        lr=2e-3,
    )

    if n_gpus > 1:
        torch.distributed.barrier()

    if proc_id == 0:
        start = time.time()

    for epoch in range(epochs):
        if n_gpus > 1:
            train_sampler.set_epoch(epoch)
        model.train()

        data_iter = train_dataloader
        if proc_id == 0:
            data_iter = tqdm(
                train_dataloader,
                desc="epoch %d" % (epoch),
                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            )
            avg_loss = 0.0

        for i, (block, head_invmap, tails, block_types) in enumerate(data_iter):
            optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            # embs: [batch_size, edge_type_count, embedding_size]
            block_types = block_types.to(dev_id)
            embs = model(block[0].to(dev_id))[head_invmap]
            embs = embs.gather(
                1,
                block_types.view(-1, 1, 1).expand(
                    embs.shape[0], 1, embs.shape[2]
                ),
            )[:, 0]
            loss = nsloss(
                block[0].dstdata[dgl.NID][head_invmap].to(dev_id),
                embs,
                tails.to(dev_id),
            )
            loss.backward()
            optimizer.step()
            sparse_optimizer.step()

            if proc_id == 0:
                avg_loss += loss.item()

                post_fix = {
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.set_postfix(post_fix)

        if n_gpus > 1:
            torch.distributed.barrier()

        if proc_id == 0:
            model.eval()
            # {'1': {}, '2': {}}
            final_model = dict(
                zip(edge_types, [dict() for _ in range(edge_type_count)])
            )
            for i in range(num_nodes):
                train_inputs = (
                    torch.tensor([i for _ in range(edge_type_count)])
                    .unsqueeze(1)
                    .to(dev_id)
                )  # [i, i]
                train_types = (
                    torch.tensor(list(range(edge_type_count)))
                    .unsqueeze(1)
                    .to(dev_id)
                )  # [0, 1]
                pairs = torch.cat(
                    (train_inputs, train_inputs, train_types), dim=1
                )  # (2, 3)
                (
                    train_blocks,
                    train_invmap,
                    fake_tails,
                    train_types,
                ) = neighbor_sampler.sample(pairs.cpu())

                node_emb = model(train_blocks[0].to(dev_id))[train_invmap]
                node_emb = node_emb.gather(
                    1,
                    train_types.to(dev_id)
                    .view(-1, 1, 1)
                    .expand(node_emb.shape[0], 1, node_emb.shape[2]),
                )[:, 0]

                for j in range(edge_type_count):
                    final_model[edge_types[j]][index2word[i]] = (
                        node_emb[j].cpu().detach().numpy()
                    )

            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == "all" or edge_types[
                    i
                ] in args.eval_type.split(","):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        valid_true_data_by_edge[edge_types[i]],
                        valid_false_data_by_edge[edge_types[i]],
                        num_workers,
                    )
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        testing_true_data_by_edge[edge_types[i]],
                        testing_false_data_by_edge[edge_types[i]],
                        num_workers,
                    )
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print("valid auc:", np.mean(valid_aucs))
            print("valid pr:", np.mean(valid_prs))
            print("valid f1:", np.mean(valid_f1s))

    if proc_id == 0:
        end = time.time()
        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)
        print("Overall ROC-AUC:", average_auc)
        print("Overall PR-AUC", average_pr)
        print("Overall F1:", average_f1)
        print("Training Time", end - start)


def train_model(network_data):
    index2word, vocab, type_nodes = generate_vocab(network_data)

    edge_types = list(network_data.keys())
    num_nodes = len(index2word)
    edge_type_count = len(edge_types)

    devices = list(map(int, args.gpu.split(",")))
    n_gpus = len(devices)
    neighbor_samples = args.neighbor_samples
    num_workers = args.workers

    g = get_graph(network_data, vocab)
    all_walks = []
    for i in range(edge_type_count):
        nodes = torch.LongTensor(type_nodes[i] * args.num_walks)
        traces, types = dgl.sampling.random_walk(
            g, nodes, metapath=[edge_types[i]] * (neighbor_samples - 1)
        )
        all_walks.append(traces)

    train_pairs = generate_pairs(all_walks, args.window_size, num_workers)
    data = g, train_pairs, index2word, edge_types, num_nodes, edge_type_count

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        mp.spawn(run, args=(n_gpus, args, devices, data), nprocs=n_gpus)


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    setup_seed(1234)

    training_data_by_type = load_training_data(file_name + "/train.txt")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/test.txt"
    )

    train_model(training_data_by_type)
