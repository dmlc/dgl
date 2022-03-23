import argparse
import dgl
import dgl.function as fn
import os
import torch
import torch.nn.functional as F

from config import CONFIG
from dgl.dataloading import SAINTSampler, DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import GCNNet
from utils import evaluate, save_log_dir, load_data

def main(args, task):
    multilabel_data = {'ppi', 'yelp', 'amazon'}
    multilabel = args.dataset in multilabel_data

    # load and preprocess dataset
    data = load_data(args, multilabel)
    g = data.g
    train_g = dgl.node_subgraph(g, data.train_nid)
    train_g.ndata['D_norm'] = 1. / train_g.in_degrees().clamp(min=1).unsqueeze(1)
    g.ndata['D_norm'] = 1. / g.in_degrees().clamp(min=1).unsqueeze(1)

    print('Pre-computing normalization coefficients')
    sampler = SAINTSampler(mode=args.sampler, budget=args.budget)
    loader = DataLoader(train_g, torch.arange(1000), sampler, num_workers=args.num_workers)

    num_train_nodes = train_g.num_nodes()
    node_count = torch.zeros(num_train_nodes)
    edge_count = torch.zeros(train_g.num_edges())

    target_num_nodes = num_train_nodes * args.norm_ratio
    num_sgs = 0
    num_sampled_nodes = 0
    pbar = tqdm(total=target_num_nodes)

    while num_sampled_nodes < target_num_nodes:
        for sg in loader:
            node_count[sg.ndata[dgl.NID]] += 1
            edge_count[sg.edata[dgl.EID]] += 1
            num_sampled_nodes += sg.num_nodes()
            pbar.update(sg.num_nodes())
            num_sgs += 1
    pbar.close()

    train_g.ndata['n_c'] = node_count.clamp(min=1)
    train_g.edata['e_c'] = edge_count.clamp(min=1)
    train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'w'))
    train_g.ndata['l_n'] = num_sgs / train_g.ndata['n_c'] / num_train_nodes

    # reconstruct sampler and dataloader for pre-fetching
    sampler = SAINTSampler(mode=args.sampler, budget=args.budget,
                           prefetch_ndata=['feat', 'D_norm', 'label', 'l_n'],
                           prefetch_edata=['w'])
    loader = DataLoader(train_g, torch.arange(num_sgs), sampler, num_workers=args.num_workers)

    # set device for dataset tensors
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    eval_device = torch.device('cpu') if args.dataset == 'amazon' else device

    model = GCNNet(
        in_dim=g.ndata['feat'].shape[1],
        hid_dim=args.n_hidden,
        out_dim=data.num_classes,
        arch=args.arch,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # logger and so on
    log_dir = save_log_dir(args)

    best_f1 = -1

    for epoch in range(args.n_epochs * num_sgs):
        print('epoch: ', epoch)
        model.train()
        for subg in loader:
            subg = subg.to(device)
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

        val_f1_mic, val_f1_mac = evaluate(eval_device, model, g, 'val_mask', multilabel)
        print("Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
        if val_f1_mic > best_f1:
            best_f1 = val_f1_mic
            print('new best val f1:', best_f1)
            torch.save(model.state_dict(), os.path.join(
                log_dir, 'best_model_{}.pkl'.format(task)))

    # test
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model_{}.pkl'.format(task))))
    test_f1_mic, test_f1_mac = evaluate(eval_device, model, g, 'test_mask', multilabel)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--task", type=str, default="ppi_n", help="type of tasks",
                        choices=list(CONFIG.keys()))
    parser.add_argument("-nw", "--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = argparse.Namespace(**CONFIG[args.task])
    config.num_workers = args.num_workers

    main(config, task=args.task)
