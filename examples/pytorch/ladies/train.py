import torch
import torch.nn.functional as F
import dgl
import tqdm
from model import *
from ladies import *

def compute_acc(pred, label):
    return (pred.argmax(1) == label).mean()

def train(g, n_classes, args):
    in_feats = g.ndata['features'].shape[1]
    device = args['device']
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    g.edata['weight'] = normalized_laplacian_edata(g)

    num_nodes = [int(n) for n in args['num_nodes'].split(',')]
    sampler = LADIESNeighborSampler(num_nodes, weight='weight', out_weight='w', replace=False)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'])
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        val_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'])

    model = Model(in_feats, args['hidden_dim'], n_classes, len(num_nodes))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'])

    for epoch in range(args['num_epochs']):
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, seeds, blocks) in enumerate(tq):
                blocks = [block.to(device) for block in blocks]
                batch_inputs = blocks[0].srcdata['features']
                batch_labels = blocks[-1].dstdata['label']

                batch_pred = model(blocks, batch_inputs)
                loss = F.cross_entropy(batch_pred, batch_labels)
                acc = compute_acc(batch_pred, batch_labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tq.set_postfix({'loss': '%.06f' % loss.item(), 'acc': '%.03f' % acc.item()})

        with tqdm.tqdm(val_dataloader) as tq:
            all_labels = []
            all_pred = []
            for step, (input_nodes, seeds, blocks) in enumerate(tq):
                blocks = [block.to(device) for block in blocks]
                batch_inputs = blocks[0].srcdata['features']
                batch_labels = blocks[-1].dstdata['labels']

                batch_pred = model(blocks, batch_inputs)

                all_labels.append(batch_labels)
                all_pred.append(batch_pred)

            all_labels = torch.cat(all_labels, 0)
            all_pred = torch.cat(all_pred, 0)
            print('Val Acc', compute_acc(all_pred, all_labels))


if __name__ == '__main__':
    g = dgl.graph((torch.randint(0, 2000, (10000,)), torch.randint(0, 2000, (10000,))))
    g = dgl.to_simple(dgl.add_reverse_edges(g), return_counts=None)
    g.ndata['features'] = torch.randn(2000, 15)
    g.ndata['label'] = torch.randint(0, 5, (2000,))
    g.ndata['mask'] = torch.randint(0, 10, (2000,))
    g.ndata['train_mask'] = g.ndata['mask'] < 8
    g.ndata['val_mask'] = g.ndata['mask'] == 8
    g.ndata['test_mask'] = g.ndata['mask'] == 9

    args = {
        'num_epochs': 1,
        'num_workers': 0,
        'batch_size': 10,
        'hidden_dim': 8,
        'lr': 1e-4,
        'num_nodes': '20,20,20',
        'device': 'cpu'}
    train(g, 5, args)
