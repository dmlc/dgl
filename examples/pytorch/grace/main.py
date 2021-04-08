import argparse

from model import Grace
from aug import aug
from dataset import load

import torch as th
import torch.nn as nn

import yaml
from yaml import SafeLoader

from eval import label_classification
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=str, default='random')
args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


if __name__ == '__main__':

    # Step 1: Load hyper-parameters =================================================================== #
    config = 'config.yaml'
    config = yaml.load(open(config), Loader=SafeLoader)[args.dataname]
    lr = config['learning_rate']
    hid_dim = config['num_hidden']
    out_dim = config['num_proj_hidden']

    num_layers = config['num_layers']
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[config['activation']]

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']

    temp = config['tau']
    epochs = config['num_epochs']
    wd = config['weight_decay']

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, num_class, train_mask, val_mask, test_mask = load(args.dataname)

    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training ======================================================================= #
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final Evaluation ===")
    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    feat = feat.to(args.device)
    embeds = model.get_embedding(graph, feat)

    '''Evaluation Embeddings  '''
    label_classification(embeds, labels, train_mask, test_mask, split=args.split, ratio=0.1)