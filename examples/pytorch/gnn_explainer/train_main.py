#-*- coding:utf-8 -*-


# The training codes of the dummy model


import argparse
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dgl.data import AIFBDataset
from .models import dummy_gnn_model
from .gengraph import *

def main(args):
    # check dataset
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'syn1':
        graph, role_ids, name = gen_syn1()
    elif args.dataset == 'syn2':
        graph, label, name = gen_syn2()
    elif args.dataset == 'syn3':
        graph, role_ids, name = gen_syn3()
    elif args.dataset == 'syn4':
        graph, role_ids, name = gen_syn4()
    elif args.dataset == 'syn5':
        graph, role_ids, name = gen_syn5()
    else:
        raise ValueError()

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy model training')
    parser.add_argument('--dataset', type=str, default='sync1', help='dataset used for training the model')

    args = parser.parse_args()
    print(args)

    main(args)