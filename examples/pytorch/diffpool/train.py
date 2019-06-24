import os
import numpy as np
import torch
import dgl
import networkx as nx
import argparse
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import tu

from model.encoder import DiffPool

def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='DiffPool arguments')
    parser.add_argument('--dataset', dest='dataset', help='Input Dataset')
    parser.add_argument('--bmname', dest='benchmark name', help='Name of the\
                        benchmark dataset')
    parser.add_argument('--pool_ratio', dest='pool_ratio', type=float, help='pooling ratio')
    parser.add_argument('--num_pool', dest='num_pool', type=int, help='num_pooling layer')
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
    parser.add_argument('--save_dir', dest='save_dir', help='model saving directory: SAVE_DICT/DATASET')
    parser.add_argument('--load_epoch', dest='load_epoch', help='load trained model params from\
                         SAVE_DICT/DATASET/model-LOAD_EPOCH')
    parser.add_argument('--data_mode', dest='data_mode', help='data preprocessing mode')

    parser.set_defaults(dataset='DD',
                        bmname='PH',
                        pool_ratio=0.15,
                        num_pool=1,
                        linkpred=True,
                        cuda=1,
                        lr=1e-3,
                        clip=2.0,
                        batch_size=29,
                        epoch=4000,
                        train_ratio=0.7,
                        test_ratio=0.1,
                        n_worker=1,
                        feature_type='default',
                        gc_per_block=3,
                        dropout=0.0,
                        method='diffpool',
                        bn=True,
                        bias=True,
                        save_dir="./model_param",
                        load_epoch=-1,
                        data_mode = 'default')
    return parser.parse_args()

def prepare_data(dataset, prog_args, fold=-1, pre_process=None):
    '''
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    '''
    if fold == -1 or fold == 0:
        shuffle = True
    else:
        shuffle = False
    
    if pre_process:
        pre_process(dataset, prog_args)

    dataset.set_fold(fold)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=prog_args.batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn,
                                       num_workers=prog_args.n_worker)

def one_hotify(labels, pad=-1):
        '''
        cast label to one hot vector
        '''
        num_instances = len(labels)
        if pad <= 0:
            dim_embedding = np.max(labels) + 1 #zero-indexed assumed
        else:
            assert pad > 0, "result_dim for padding one hot embedding not set!"
            dim_embedding = pad + 1
        embeddings = np.zeros((num_instances, dim_embedding))
        embeddings[np.arange(num_instances), labels] = 1

        return embeddings


def pre_process(dataset, prog_args):
        """
        diffpool specific data partition, pre-process and shuffling
        """
        if dataset.data_mode != "default":
            print("overwrite node attributes with DiffPool's preprocess setting")
            if prog_args.data_mode == 'id':
                for g in dataset.graph_lists:
                    id_list = np.arange(g.number_of_nodes())
                    g.ndata['feat'] = one_hotify(id_list, pad=dataset.max_num_node)
            elif prog_args.data_mode == 'deg-num':
                for g in dataset.graph_lists:
                    g.ndata['feat'] = np.expand_dims(g.in_degrees(), axis=1)

            elif prog_args.data_mode == 'deg':
                # max degree is disabled.
                for g in dataset.graph_lists:
                    degs = list(g.in_degrees())
                    degs_one_hot = one_hotify(degs, pad=dataset.max_degrees)
                    g.ndata['feat'] = degs_one_hot
        """
        elif self.kwargs['feature_mode'] == 'struct':
            for g in self.graph_lists:
                degs = list(g.in_degrees())
                degs_one_hot = self.one_hotify(degs, pad=True, result_dim=self.max_degrees)
                nxg = g.to_networkx().to_undirected()
                clustering_coeffs = np.array(list(nx.clustering(nxg).values()))
                clustering_embedding = np.expand_dims(clustering_coeffs,
                                                      axis=1)
                struct_feats = np.concatenate((degs_one_hot,
                                               clustering_embedding),
                                              axis=1)
                if self.use_node_attr:
                    g.ndata['feat'] = np.concatenate((struct_feats,
                                                      g.ndata['feat']),
                                                     axis=1)
                else:
                    g.ndata['feat'] = struct_feats

        assert 'feat' in self.graph_lists[0].ndata, "node feature not initialized!"

        if self.kwargs['assign_feat'] == 'id':
            for g in self.graph_lists:
                id_list = np.arange(g.number_of_nodes())
                g.ndata['a_feat'] = self.one_hotify(id_list, pad=True,
                                                    result_dim=self.max_num_node)
        else:
            for g in self.graph_lists:
                id_list = np.arange(g.number_of_nodes())
                id_embedding = self.one_hotify(id_list, pad=True,
                                               result_dim=self.max_num_node)
                g.ndata['a_feat'] = np.concatenate((id_embedding,
                                                    g.ndata['feat']),
                                                   axis=1)
        """
        # sanity check
        assert dataset.graph_lists[0].ndata['feat'].shape[1] ==\
                dataset.graph_lists[1].ndata['feat'].shape[1]


def graph_classify_task(prog_args):
    '''
    perform graph classification task
    '''
    diffpool_kw_args = {}
    diffpool_kw_args['feature_mode'] = prog_args.feature_type
    diffpool_kw_args['assign_feat'] = 'id'
    use_node_attr = False
    if prog_args.dataset == 'ENZYMES':
        use_node_attr = True
    dataset = tu.TUDataset(name=prog_args.dataset, n_split=3, split_ratio=[0.8, 0.1, 0.1])
    dataset_val = tu.TUDataset(name=prog_args.dataset, n_split=3, split_ratio=[0.8, 0.1, 0.1])
    dataset_test = tu.TUDataset(name=prog_args.dataset, n_split=3, split_ratio=[0.8, 0.1, 0.1])
    train_dataloader = prepare_data(dataset, prog_args, fold=0,
                                    pre_process=pre_process)
    val_dataloader = prepare_data(dataset_val, prog_args, fold=1,
                                  pre_process=pre_process)
    test_dataloader = prepare_data(dataset_test, prog_args,
                                   fold=2,pre_process=pre_process)
    input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))
    assert len(dataset) % prog_args.batch_size == 0, "training set not divisible by batch size"
    # assert len(dataset_val) % prog_args.batch_size == 0, "val set not divisible by batch size"
    # assert len(dataset_test) % prog_args.batch_size == 0, "test set not divisible by batch size"

    hidden_dim = 64 # used to be 64
    embedding_dim = 64

    # calculate assignment dimension: pool_ratio * largest graph's maximum
    # number of nodes  in the dataset
    assign_dim = int(max_num_node * prog_args.pool_ratio) * prog_args.batch_size
    print("++++++++++MODEL STATISTICS++++++++")
    print("model hidden dim is", hidden_dim)
    print("model embedding dim for graph instance embedding", embedding_dim)
    print("initial batched pool graph dim is", assign_dim)
    activation = F.relu
    

    # initialize model
    # 'diffpool' : diffpool
    model = DiffPool(input_dim, 
                     hidden_dim, 
                     embedding_dim,  
                     label_dim,
                     activation, 
                     prog_args.gc_per_block,  
                     prog_args.dropout, 
                     prog_args.num_pool, 
                     prog_args.linkpred,
                     prog_args.batch_size, 
                     'maxpool', 
                     assign_dim,
                     prog_args.pool_ratio)
    
    if prog_args.load_epoch >= 0 and prog_args.save_dir is not None:
        model.load_state_dict(torch.load(prog_args.save_dir + "/" + prog_args.dataset\
                                         + "/model.iter-" + str(prog_args.load_epoch)))

    print("model init finished")
    print("MODEL:::::::", prog_args.method)
    if prog_args.cuda:
        model = model.cuda()
    
    logger = train(train_dataloader, model, prog_args, val_dataset=val_dataloader)
    result = evaluate(test_dataloader, model, prog_args, logger)
    print("test  accuracy {}%".format(result*100))

def collate_fn(batch):
    '''
    collate_fn for dataset batching
    transform ndata to tensor (in gpu is available)
    '''
    graphs, labels = map(list, zip(*batch))
    #cuda = torch.cuda.is_available()

    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value)
    batched_graphs = dgl.batch(graphs)

    # move to cuda
    #for (key, value) in batched_graphs.ndata.items():
    #    if cuda:
    #        batched_graphs.ndata[key] = value.cuda()
    #    else:
    #        batched_graphs.ndata[key] = value

    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))

    # move to cuda
    #if cuda:
    #    batched_labels = batched_labels.cuda()
    return batched_graphs, batched_labels

def train(dataset, model, prog_args, same_feat=True, val_dataset=None):
    '''
    training function
    '''
    dir = prog_args.save_dir + "/" + prog_args.dataset
    if not os.path.exists(dir):
        os.makedirs(dir)
    dataloader = dataset
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=0.001)
    early_stopping_logger = {"best_epoch":-1, "val_acc": -1}

    if prog_args.cuda > 0:
        torch.cuda.set_device(0)
    for epoch in range(prog_args.epoch):
        begin_time = time.time()
        model.train()
        train_accu = 0
        print("EPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                graph_labels = graph_labels.cuda()

                
            model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels).item()
            train_accu += correct
            loss = model.loss(ypred, graph_labels)
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)
            optimizer.step()


        train_accu = train_accu / (len(dataloader)*prog_args.batch_size)
        print("train accuracy for this epoch {} is {}%".format(epoch,
                                                              train_accu*100))
        elapsed_time = time.time() - begin_time
        print("loss {} with epoch time {} s & computation time {} s ".format(loss.item(), elapsed_time, computation_time))
        if val_dataset is not None:
            result = evaluate(val_dataset, model, prog_args)
            print("validation  accuracy {}%".format(result*100))
            if result >= early_stopping_logger['val_acc'] and result <=\
            train_accu:
                early_stopping_logger.update(best_epoch=epoch, val_acc=result)
                if prog_args.save_dir is not None:
                    torch.save(model.state_dict(), prog_args.save_dir + "/" + prog_args.dataset\
                                         + "/model.iter-" + str(early_stopping_logger['best_epoch']))
            print("best epoch is EPOCH {}, val_acc is {}%".format(early_stopping_logger['best_epoch'], 
                                                                early_stopping_logger['val_acc']*100))
        torch.cuda.empty_cache()
    return early_stopping_logger

def evaluate(dataloader, model, prog_args, logger=None):
    '''
    evaluate function
    '''
    if logger is not None and prog_args.save_dir is not None:
        model.load_state_dict(torch.load(prog_args.save_dir + "/" + prog_args.dataset\
                                         + "/model.iter-" + str(logger['best_epoch'])))
    model.eval()
    correct_label = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                graph_labels = graph_labels.cuda()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi==graph_labels)
            correct_label += correct.item()
    result = correct_label / (len(dataloader)*prog_args.batch_size)
    return result

def main():
    '''
    main
    '''
    prog_args = arg_parse()
    print(prog_args)
    graph_classify_task(prog_args)


if __name__ == "__main__":
    main()
