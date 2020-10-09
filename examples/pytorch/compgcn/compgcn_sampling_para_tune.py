#-*- coding:utf-8 -*-

# Author:james Zhang
# Datetime:20-10-09 10:05
# Project: DGL
"""
    This file is designed for hyper-parameter tuning to achieve the best accuary performance in 1 DGL dataset:
    - AM: The AM dataset is too big to fit in one GPU and too slow to tune in CPU

    1. Basically use early stop to get the best val-performance's hyper-parameters, but
    2. Because this AM dataset might need run a couple runs before going to the lowest optimal points.

    3. Only support 1 GPU to simplify training for best performance.

    Main hyper-parameters would invovle:
    0. The composition function: SUB, MUL, and CCORR
    1. Vector Basis for edge type;
    2. The number of layers of the CompGCN model;
    3. The number of hidden dimensionality;
    4. The dropout rate

"""

import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dgl.data import AMDataset, MUTAGDataset, AIFBDataset, BGSDataset
from dgl.dataloading.neighbor import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from data_utils import build_dummy_comp_data
from models.model import CompGCN
from model_utils import early_stopper


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def main(args):

    # Step 1： Prepare graph data and split into train/validation ============================= #
    if args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError('This version is only for AM dataset. Please use \'am\' as the input argument.')

    # Load from hetero-graph
    heterograph = dataset[0]

    # number of classes to predict, and the node type
    num_classes = dataset.num_classes
    target = dataset.predict_category

    # basic information of the dataset
    num_rels = len(heterograph.canonical_etypes)

    train_mask = heterograph.nodes[target].data.pop('train_mask')
    test_mask = heterograph.nodes[target].data.pop('test_mask')
    train_idx = th.nonzero(train_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()
    labels = heterograph.nodes[target].data.pop('labels')

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    num_of_ntype = len(heterograph.ntypes)
    num_of_etype = len(heterograph.ntypes)
    print('In Dataset: {}, node types num: {}'.format(args.dataset, num_of_ntype))
    print('In Dataset: {}, edge types num: {}'.format(args.dataset, num_of_etype))

    num_basis = int(args.num_basis * num_of_etype) + 1

    # check cuda
    use_cuda = (args.gpu >= 0 and th.cuda.is_available())
    print("If use GPU: {}".format(use_cuda))

    in_feat_dict = {}
    n_feats = {}

    # For node featureless, use an additional embedding layer to transform the data
    for ntype in heterograph.ntypes:
        n_feats[ntype] = th.arange(heterograph.number_of_nodes(ntype))
        in_feat_dict[ntype] = num_rels

        # if use_cuda:
        #     n_feats[ntype] = n_feats[ntype].to('cuda:{}'.format(args.gpu))

    # if use_cuda:
    #     labels = labels.to('cuda:{}'.format(args.gpu))

    # build sampler from dataloader
    sampler = MultiLayerFullNeighborSampler(n_layers=args.num_layer - 2)
    train_dataloader = NodeDataLoader(g=heterograph,
                                      nids={target: train_idx},
                                      block_sampler=sampler,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4)
    valid_dataloader = NodeDataLoader(g=heterograph,
                                      nids={target: val_idx},
                                      block_sampler=sampler,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4)

    # Step 2: Create model =================================================================== #
    input_embs = nn.ModuleDict()
    for ntype in heterograph.ntypes:
        input_embs[ntype] = nn.Embedding(heterograph.number_of_nodes(ntype), num_rels)

    compgcn_model = CompGCN(in_feat_dict=in_feat_dict,
                            hid_dim=args.hid_dim,
                            num_layers=args.num_layer,
                            out_feat=num_classes,
                            num_basis=num_basis,
                            num_rel=num_rels,
                            comp_fn=args.comp_fn,
                            dropout=args.drop_out,
                            activation=F.relu,
                            batchnorm=True
                            )

    if use_cuda:
        # input_embs.to('cuda:{}'.format(args.gpu))
        compgcn_model = compgcn_model.to('cuda:{}'.format(args.gpu))
        # heterograph = heterograph.to('cuda:{}'.format(args.gpu))

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': input_embs.parameters(), 'lr':args.lr, 'weight_decay':5e-4},
                            {'params': compgcn_model.parameters(), 'lr':args.lr, 'weight_decay':5e-4}
                            ])

    earlystoper = early_stopper(patience=2, verbose=False, delta=0.01)

    # Step 4: training epoches =============================================================== #
    for epoch in range(200):

        # forward
        input_embs.train()
        compgcn_model.train()

        for input_nodes, seeds, block in train_dataloader:

            print(block)

            in_n_feats = extract_embed(n_feats, input_nodes)
            in_n_feats = {k : e.to('cuda:{}'.format(args.gpu)) for k, e in in_n_feats.items()}
            block = [blk.to('cuda:{}'.format(args.gpu)) for blk in block]

            logits = compgcn_model.forward(block, in_n_feats)

            # compute loss
            seeds = seeds[target]
            tr_loss = loss_fn(logits[target], labels[seeds])

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            tr_acc = th.sum(logits[target].argmax(dim=1) == labels[seeds]).item() / len(seeds)

        print("Last batch tr_acc: {:.4f} | Last batch tr_loss: {:.4f}".format(tr_acc, tr_loss.item()))

        input_embs.eval()
        compgcn_model.eval()

        for input_nodes, seeds, block in valid_dataloader:

            in_n_feats = extract_embed(n_feats, input_nodes)
            in_n_feats = {k : e.to('cuda:{}'.format(args.gpu)) for k, e in in_n_feats.items()}
            block = [blk.to('cuda:{}'.format(args.gpu)) for blk in block]

            val_loss = loss_fn(logits[target], labels[seeds])
            val_acc = th.sum(logits[target].argmax(dim=1) == labels[seeds]).item() / len(seeds)

        print("Last batch tr_acc: {:.4f} | Last batch tr_loss: {:.4f}".format(val_acc, val_loss.item()))

        if epoch > 40:
            earlystoper.earlystop(val_loss, val_acc, None)
            if earlystoper.is_earlystop:
                break

    # transfer back to cpu for testing
    input_embs = input_embs.to('cpu')
    compgcn_model = compgcn_model.to('cpu')

    input_embs.eval()
    compgcn_model.eval()

    in_n_feats = {}
    for ntype, feat in n_feats.items():
        in_n_feats[ntype] = input_embs[ntype](feat)

    logits = compgcn_model.forward(heterograph, in_n_feats)

    test_loss = loss_fn(logits[target][test_idx], labels[test_idx])
    test_acc = th.sum(logits[target][test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    return test_acc, test_loss.item(), input_embs, compgcn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoSH CompGCN Full Graph')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    # disable below arguments and
    # parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimensionalities")
    # parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    # parser.add_argument("--num_basis", type=int, default=40, help="Number of basis")
    parser.add_argument("--rev_indicator", type=str, default='_inv', help="Indicator of reversed edge")
    # parser.add_argument("--comp_fn", type=str, default='sub', help="Composition function")
    # parser.add_argument("--max_epoch", type=int, default=200, help="The max number of epoches")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    # parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out rate")
    parser.add_argument("--batch_size", type=int, default=16, required=True)
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)

    np.random.seed(123456)
    th.manual_seed(123456)

    # HP tunning ranges
    hid_dims = [4, 8, 16, 32, 64]
    num_layers = [3, 4, 5, 6]
    num_basis_factor = [1/(2**0), 1/(2**1), 1/(2**2), 1/(2**4), 1/(2**5)]
    comp_fns = ['sub', 'mul', 'ccorr']
    drop_outs = [0, 0.1, 0.2, 0.3, 0.4]

    # Run parameter tunning
    results = []

    best_test_acc = 0
    best_input_embs = None
    best_compgcn_model = None

    for drop_out in drop_outs:
        args.drop_out = drop_out
        for hid_dim in hid_dims:
            args.hid_dim = hid_dim
            for num_layer in num_layers:
                args.num_layer = num_layer
                for num_basis in num_basis_factor:
                    args.num_basis = num_basis
                    for comp_fn in comp_fns:
                        args.comp_fn = comp_fn

                        test_acc, test_loss, input_embs, compgcn_model = main(args)

                        print(drop_out, '|', hid_dim, '|', num_layer, '|', num_basis, '|', comp_fn)
                        print(test_acc, '|', test_loss)
                        print()

                        # Save results in a test result file
                        results.append([drop_out, hid_dim, num_layer, num_basis, comp_fn, test_acc, test_loss])

                        # output the best results in the current settings
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            best_input_embs = input_embs
                            best_compgcn_model = compgcn_model

    # After all parameter tunning, save the models to local files
    results_df = pd.DataFrame(data=results,
                              columns=['Drop_out', 'Hid_dim', 'Num_layer', 'Num_basis', 'Comp_fn', 'Test_acc', 'Test_loss'])
    results_df.to_csv('/data/{}.csv'.format(args.dataset), index=False)

    if best_test_acc > 0:
        # TODO: Save model here
        pass