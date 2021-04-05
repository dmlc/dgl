import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data


def main(args):

    multitask_data = set(['ppi', 'yelp', 'amazon'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    data = load_data(args, multitask)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
    train_g = g.subgraph(train_nid)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.to(args.gpu)
        train_g = train_g.to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape, ", g.ndata['feat'].shape)

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=args.batch_norm
    )

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1
    print("n tain nodes", n_train_samples)
    for epoch in range(args.n_epochs):
        model.train()
        # forward
        pred = model(train_g)
        train_labels = train_g.ndata['label']

        if multitask:
            loss = F.binary_cross_entropy_with_logits(pred, train_labels)
        else:
            loss = F.cross_entropy(pred, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch}/{args.n_epochs}: training loss", loss.item())

        # evaluate
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels, val_mask, multitask)
            print(
                "Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    # test
    if True:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac = evaluate(
        model, g, labels, test_mask, multitask)
    print("Test F1-mic{:.4f}, Test F1-mac{:.4f}".format(test_f1_mic, test_f1_mac))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # data source params
    parser.add_argument("--dataset", type=str, choices=['ppi', 'flickr', 'reddit', 'yelp', 'amazon'], default='ppi',
                        help="Name of dataset.")
    # cuda params
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU index. Default: -1, using CPU.")
    # model params
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="Number of hidden gcn units")
    parser.add_argument("--arch", type=str, default="1-0-1-0",
                        help="Network architecture. 1 means an order 1 layer (self feature plus 1-hop neighbor "
                             "feature), and 0 means an order 0 layer (self feature only)")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate")
    parser.add_argument("--batch-norm", action='store_true',
                        help="Whether to use batch norm")
    # training params
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 reg")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Number of epoch of doing inference on validation")
    parser.add_argument("--use-val", action='store_true',
                        help="Whether to use validated best model to test")
    parser.add_argument("--note", type=str, default='none',
                        help="Note for log dir")

    args = parser.parse_args()

    print(args)

    main(args)
