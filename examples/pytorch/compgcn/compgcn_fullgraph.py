#-*- coding:utf-8 -*-

# Author:james Zhang
# Datetime:20-09-03 18:32
# Project: BoSH
"""
    This file is the running entry point for full graph training with the dummy data:
    1. Only used for code testing.

"""

import argparse
import os
import dgl
import torch as th
import torch.optim as optim

from dgl.data import AMDataset, MUTAGDataset, AIFBDataset, BGSDataset
from data_utils import build_dummy_comp_data
from models.model import CompGCN


def main(args):

    # Step 1： Prepare graph data and split into train/validation ============================= #
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

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

    print(heterograph.ntypes)
    print(heterograph.etypes)

    # check cuda
    use_cuda = (args.gpu >= 0 and th.cuda.is_available())
    print("If use GPU: {}".format(use_cuda))

    if use_cuda:
        th.cuda.set_device(args.gpu)

    in_feat_dict = {}
    n_feats = {}
    for ntype in heterograph.ntypes:
        n_feats[ntype] = th.arange(heterograph.number_of_nodes(ntype)).view(-1, 1).float()
        in_feat_dict[ntype] = th.tensor(1)

        if use_cuda:
            n_feats[ntype] = n_feats[ntype].cuda()
            print(n_feats[ntype].device)
            in_feat_dict[ntype] = in_feat_dict[ntype].cuda()
            print(in_feat_dict[ntype].device)

    if use_cuda:
        labels = labels.cuda()

    # Step 2: Create model =================================================================== #
    compgcn_model = CompGCN(in_feat_dict=in_feat_dict,
                            hid_dim=args.hid_dim,
                            num_layers=args.num_layers,
                            out_feat=num_classes,
                            num_basis=args.num_basis,
                            num_rel=num_rels,
                            comp_fn=args.comp_fn,
                            dropout=0.0,
                            activation=None,
                            batchnorm=False
                            )

    if use_cuda:
        compgcn_model.cuda()
        heterograph = heterograph.to(th.device('cuda:{}'.format(args.gpu)))

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(compgcn_model.parameters(), lr=0.005, weight_decay=5e-4)

    # Step 4: training epoches =============================================================== #
    for epoch in range(args.max_epoch):
        # 前向传播
        compgcn_model.train()
        logits = compgcn_model.forward(heterograph, n_feats)

        # 计算loss
        tr_loss = loss_fn(logits[target][train_idx], labels[train_idx])
        val_loss = loss_fn(logits[target][val_idx], labels[val_idx])

        # 反向传播
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        train_acc = th.sum(logits[target][train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_acc = th.sum(logits[target][val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, tr_loss.item(), val_acc, val_loss.item()))

    print()

    compgcn_model.eval()
    logits = compgcn_model.forward(heterograph, n_feats)
    logits = logits[target][test_idx]
    test_loss = loss_fn(logits[target][test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    # Step 5: If need, save model to file ============================================================== #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoSH CompGCN Full Graph')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimensionalities")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--num_basis", type=int, default=50, help="Number of basis")
    parser.add_argument("--rev_indicator", type=str, default='_inv', help="Indicator of reversed edge")
    parser.add_argument("--comp_fn", type=str, default='sub', help="Composition function")
    parser.add_argument("--max_epoch", type=int, default=100, help="The max number of epoches")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)