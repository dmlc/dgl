import numpy as np
import torch
import dgl
import networkx as nx
import argparse
import os
import random
import time
import sklearn.metrics as metrics

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import tu

from model.encoder import GraphEncoder, DiffPoolEncoder

def arg_parse():
    parser = argparse.ArgumentParser(description='DiffPool arguments')
    parser.add_argument('--dataset', dest='dataset', help='Input Dataset')
    parser.add_argument('--bmname', dest='benchmark name', help='Name of the benchmark datset')
    parser.add_argument('--pool_ratio', dest='pool_ratio', type=float, help='pooling ratio')
    parser.add_argument('--num_pool', dest='num_pool', type=int,  help='num_pooling layer')
    parser.add_argument('--link_pred', dest='linkpred', action='store_const',
                        const=True, default=True,
                        help='switch of link prediction object')
    parser.add_argument('--cuda', dest='cuda', type=int, help='switch cuda')
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate')
    parser.add_argument('--clip', dest='clip', type=float, help='gradient clipping')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', dest='epoch', type=int,
                        help='num-of-epoch')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='ratio of trainning dataset split')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='ratio of testing dataset split')
    parser.add_argument('--num_workers', dest='n_worker', type=int,
                        help='number of workers when dataloading')
    parser.add_argument('--feature', dest='feature_type',
                        help='feature type, could be id or deg')
    parser.add_argument('--gc-per-block', dest='gc_per_block', type=int,
                        help='number of graph conv layer per block')
    parser.add_argument('--bn', dest='bn', action='store_const', const=True,
                        default=True, help='switch for bn')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='dropout rate')
    parser.add_argument('--bias', dest='bias', action='store_const',
                        const=True, default=True, help='switch for bias')

    parser.set_defaults(dataset='ENZYMES',
                        bmname='PH',
                        pool_ratio=0.25,
                        num_pool=1,
                        linkpred=True,
                        cuda=1,
                        lr=0.001,
                        clip=2.0,
                        batch_size=10,
                        epoch=200,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        n_worker=0,
                        feature_type='default',
                        gc_per_block=3,
                        dropout=0.0,
                        method='diffpool',
                        bn=False,# \TODO batch normalization is not enabled
                        # We batch graphs differently.
                        bias=True)
    return parser.parse_args()

def prepare_data(dataset, prog_args, mode):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset.set_mode(mode)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=prog_args.batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn,
                                       num_workers=prog_args.n_worker)


def graph_classify_task(prog_args):
    diffpool_kw_args = {}
    diffpool_kw_args['feature_mode'] = prog_args.feature_type
    diffpool_kw_args['assign_feat'] = 'id'
    use_node_attr = False
    if prog_args.dataset == 'ENZYMES':
        use_node_attr = True
    dataset = tu.DiffpoolDataset(name=prog_args.dataset,
                                 use_node_attr=use_node_attr,
                                 use_node_label=True,mode='train',
                                 train_ratio=prog_args.train_ratio,
                                 test_ratio=prog_args.test_ratio,
                                 **diffpool_kw_args)

    val_dataset = tu.DiffpoolDataset(name=prog_args.dataset,
                                     use_node_attr=use_node_attr,
                                     use_node_label=True,mode='val',
                                     train_ratio=prog_args.train_ratio,
                                     test_ratio=prog_args.test_ratio,
                                     **diffpool_kw_args)

    test_dataset = tu.DiffpoolDataset(name=prog_args.dataset,
                                      use_node_attr=use_node_attr,
                                      use_node_label=True,mode='test',
                                      train_ratio=prog_args.train_ratio,
                                      test_ratio=prog_args.test_ratio,
                                      **diffpool_kw_args)





    input_dim, assign_input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset assign_feature input dimension is (if used)",
          assign_input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    hidden_dim = 64
    embedding_dim = 64
    pred_hidden_dims = [64,64]
    assign_dim = int(max_num_node * prog_args.pool_ratio) * prog_args.batch_size
    print("++++++++++MODEL STATISTICS++++++++")
    print("model hidden dim is", hidden_dim)
    print("model embedding dim fr graph instance embedding", embedding_dim)
    print("initial batched pool graph dim is", assign_dim)


    train_dataloader = prepare_data(dataset, prog_args, mode='train')
    val_dataloader = prepare_data(val_dataset, prog_args, mode='val')
    test_dataloader = prepare_data(test_dataset, prog_args, mode='test')
    activation = F.relu

    # initialize model
    if prog_args.method == 'base':
        basekwargs = {'concat':True, 'bn':prog_args.bn, 'bias':True,
                      'aggregator_type':'maxpool'}
        model = GraphEncoder(input_dim, hidden_dim, embedding_dim,
                             pred_hidden_dims, label_dim, activation,
                             prog_args.gc_per_block, prog_args.dropout, **basekwargs)
        if prog_args.cuda:
            model = model.cuda()
        print("model init finished")
        print("MODEL:::::::::", prog_args.method)
    elif prog_args.method == 'diffpool':
        diffpoolkwargs = {'concat':True, 'bn':prog_args.bn, 'bias':True,
                      'aggregator_type':'maxpool', 'pool_ratio':
                      prog_args.pool_ratio, 'assign_dim': assign_dim,
                      'batch_size': prog_args.batch_size}
        assign_input_dim = -1
        assign_n_layers = -1
        assign_hidden_dim = hidden_dim
        model = DiffPoolEncoder(input_dim, assign_input_dim, hidden_dim, embedding_dim,
                                pred_hidden_dims, assign_hidden_dim, label_dim, activation,
                                prog_args.gc_per_block, assign_n_layers, prog_args.dropout,
                                prog_args.num_pool, prog_args.linkpred,
                                **diffpoolkwargs)
        print("model init finished")
        print("MODEL:::::::", prog_args.method)
        if prog_args.cuda:
            model = model.cuda()

    train(train_dataloader, model, prog_args, val_dataset=val_dataloader)
    result = evaluate(test_dataloader, model, prog_args)
    print("test  accuracy {}".format(result))

def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    cuda = torch.cuda.is_available()
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value)
    batched_graphs = dgl.batch(graphs)
    for (key, value) in batched_graphs.ndata.items():
        if cuda:
            batched_graphs.ndata[key] = value.cuda()
        else:
            batched_graphs.ndata[key] = value

    batched_labels = torch.LongTensor(np.concatenate(labels, axis=0))
    if cuda:
        batched_labels = batched_labels.cuda()

    return batched_graphs, batched_labels

def train(dataset, model, prog_args, same_feat=True, val_dataset=None):
    dataloader = dataset
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad,
                                        model.parameters()), lr=0.001)
    # iter
    best_val_result = {'epoch': 0, 'loss': 0, 'acc': 0}
    test_result = {'epoch': 0, 'loss': 0, 'acc': 0}

    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    if prog_args.cuda > 0:
        torch.cuda.set_device(0)
    else:
        cuda = False



    for epoch in range(prog_args.epoch):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        train_accu = 0
        print("EPOCH ###### {} ######".format(epoch))
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            model.zero_grad()


            ypred = model(batch_graph)

            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels).item()
            train_accu += correct
            if prog_args.method == 'base':
                loss = model.loss(ypred, graph_labels)
                loss.backward()
            elif prog_args.method == 'diffpool':
                loss = model.loss(ypred, graph_labels)
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)
            optimizer.step()


        train_accu = train_accu / (len(dataloader)*prog_args.batch_size)
        print("train accuracy for this epoch {} is {}".format(epoch,
                                                              train_accu))
        elapsed_time = time.time() - begin_time
        print("loss {} with epoch time {} s ".format(loss.item(), elapsed_time))
        if val_dataset is not None:
            result = evaluate(val_dataset, model, prog_args)
            print("validation  accuracy {}".format(result))
        torch.cuda.empty_cache()

def evaluate(dataloader, model, prog_args):
    model.eval()
    indi_list = []
    preds = []
    gt_labels = []
    correct_label = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi==graph_labels)
            correct_label += correct.item()
    result = correct_label / (len(dataloader)*prog_args.batch_size)
    return result

def main():
    #torch.multiprocessing.set_start_method('spawn')
    # Not supported by DGL yet!
    prog_args = arg_parse()
    print(prog_args)
    graph_classify_task(prog_args)


if __name__ == "__main__":
    main()
