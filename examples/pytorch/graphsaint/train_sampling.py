import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from sampler_jiahanli import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from config import CONFIG
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data, calc_f1

import warnings
# TODO: why author's method does not work here?
def main(args):
    warnings.filterwarnings('ignore')
    multilabel_data = {'ppi', 'yelp', 'amazon'}
    multilabel = args.dataset in multilabel_data

    # load and preprocess dataset
    data = load_data(args, multilabel)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = data.train_nid

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    # load sampler

    kwargs = {
        'dn': args.dataset, 'g': g, 'train_nid': train_nid, 'num_workers': args.num_workers, 'train': False,
        'num_subg_train': args.num_subg_train, 'num_subg_norm': args.num_subg_norm,
        'batch_size_norm': args.batch_size_norm, 'online': args.online, 'num_repeat': args.num_repeat
    }


    # for _ in range(args.test_repeat): # TODO: TEST
    if args.sampler == "node":
        saint_sampler = SAINTNodeSampler(args.node_budget, **kwargs)
    elif args.sampler == "edge":
        saint_sampler = SAINTEdgeSampler(args.edge_budget, **kwargs)
    elif args.sampler == "rw":
        saint_sampler = SAINTRandomWalkSampler(args.num_roots, args.length, **kwargs)
    else:
        raise NotImplementedError

    # return # TODO: TEST, this flag is set to measure pre-sampling time without time-consuming training

    saint_sampler.train = True
    loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
                        shuffle=True, num_workers=0, drop_last=False)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape:", g.ndata['feat'].shape)

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        aggr=args.aggr
    )

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("GPU memory allocated before training(MB)",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1

    for epoch in range(args.n_epochs):
        # for j, subg in enumerate(subg_iter):
        # t = time.perf_counter()
        for j, subg in enumerate(loader):
            # print("Sampling time per iter (one subgraph) in training: {}".format(time.perf_counter() - t))
            # sync with upper level training graph
            if cuda:
                subg = subg.to(torch.cuda.current_device())
            model.train()
            # forward
            pred = model(subg)
            batch_labels = subg.ndata['label']

            if multilabel:
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels, reduction='sum',
                                                          weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction='none')
                loss = (subg.ndata['l_n'] * loss).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            if j == len(loader) - 1:
                model.eval()
                with torch.no_grad():
                    train_f1_mic, train_f1_mac = calc_f1(batch_labels.cpu().numpy(),
                                                         pred.cpu().numpy(), multilabel)
                    print(f"epoch:{epoch + 1}/{args.n_epochs}, Iteration {j + 1}/"
                          f"{len(loader)}:training loss", loss.item())
                    print("Train F1-mic {:.4f}, Train F1-mac {:.4f}".format(train_f1_mic, train_f1_mac))

        # evaluate
        model.eval()
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels, val_mask, multilabel)
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
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac = evaluate(
        model, g, labels, test_mask, multilabel)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='GraphSAINT')
    # # data source params
    # parser.add_argument("--dataset", type=str, choices=['ppi', 'flickr'], default='ppi',
    #                     help="Name of dataset.")
    #
    # # cuda params
    # parser.add_argument("--gpu", type=int, default=-1,
    #                     help="GPU index. Default: -1, using CPU.")
    #
    # # sampler params
    # parser.add_argument("--sampler", type=str, default="node", choices=['node', 'edge', 'rw'],
    #                     help="Type of sampler")
    # parser.add_argument("--node-budget", type=int, default=6000,
    #                     help="Expected number of sampled nodes when using node sampler")
    # parser.add_argument("--edge-budget", type=int, default=4000,
    #                     help="Expected number of sampled edges when using edge sampler")
    # parser.add_argument("--num-roots", type=int, default=3000,
    #                     help="Expected number of sampled root nodes when using random walk sampler")
    # parser.add_argument("--length", type=int, default=2,
    #                     help="The length of random walk when using random walk sampler")
    # parser.add_argument("--num-repeat", type=int, default=50,
    #                     help="Number of times of repeating sampling one node to estimate edge / node probability")
    #
    # # model params
    # parser.add_argument("--n-hidden", type=int, default=512,
    #                     help="Number of hidden gcn units")
    # parser.add_argument("--arch", type=str, default="1-0-1-0",
    #                     help="Network architecture. 1 means an order-1 layer (self feature plus 1-hop neighbor "
    #                          "feature), and 0 means an order-0 layer (self feature only)")
    # parser.add_argument("--dropout", type=float, default=0,
    #                     help="Dropout rate")
    # parser.add_argument("--no-batch-norm", action='store_true',
    #                     help="Whether to use batch norm")
    # parser.add_argument("--aggr", type=str, default="concat", choices=['mean', 'concat'],
    #                     help="How to aggregate the self feature and neighbor features")
    #
    # # training params
    # parser.add_argument("--n-epochs", type=int, default=100,
    #                     help="Number of training epochs")
    # parser.add_argument("--lr", type=float, default=0.01,
    #                     help="Learning rate")
    # parser.add_argument("--val-every", type=int, default=1,
    #                     help="Frequency of evaluation on the validation set in number of epochs")
    # parser.add_argument("--use-val", action='store_true',
    #                     help="whether to use validated best model to test")
    # parser.add_argument("--log-dir", type=str, default='none',
    #                     help="Log file will be saved to log/{dataset}/{log_dir}")
    #
    # args = parser.parse_args()
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--task", type=str, default="ppi_n", help="type of tasks")
    task = parser.parse_args().task
    args = argparse.Namespace(**CONFIG[task])
    print(args)


    main(args)
