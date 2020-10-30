#-*- coding:utf-8 -*-


# The major idea of the overall GNN model explanation


import argparse
import os

from .models import dummy_gnn_model


def main(args):
    # load an exisitng model or train a new model
    if os.path.exists(args.model_path):
        pass
    else:
        pass





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--model_path', type=str, default='./model.pd', help='The path of saved model to be explained.')
    parser.add_argument('--dataset', type=str, default='aifb', help='dataset used for training the model')

    args = parser.parse_args()
    print(args)

    main(args)