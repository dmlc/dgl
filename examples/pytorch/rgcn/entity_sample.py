"""
Differences compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import gc
import torch as th
import torch.nn.functional as F
import dgl

from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from tqdm import tqdm

from entity_utils import load_data
from model import RelGraphEmbedLayer, RGCN

def init_dataloaders(args, g, train_idx, test_idx, target_idx, device, num_gpus=0):
    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    sampler = MultiLayerNeighborSampler(fanouts)
    use_ddp = num_gpus > 0

    train_loader = NodeDataLoader(
        g,
        target_idx[train_idx],
        sampler,
        use_ddp=use_ddp,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    # The datasets do not have a validation subset, use the train subset
    val_loader = NodeDataLoader(
        g,
        target_idx[train_idx],
        sampler,
        use_ddp=use_ddp,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    test_sampler = MultiLayerNeighborSampler([None] * len(fanouts))
    test_loader = NodeDataLoader(
        g,
        target_idx[test_idx],
        test_sampler,
        use_ddp=use_ddp,
        device=device,
        batch_size=32,
        shuffle=False,
        drop_last=False)

    return train_loader, val_loader, test_loader

def init_models(args, device, node_feats, num_classes, num_rels):
    embed_layer = RelGraphEmbedLayer(device if not args.dgl_sparse else th.device('cpu'),
                                     device,
                                     node_feats,
                                     args.n_hidden,
                                     dgl_sparse=args.dgl_sparse)

    model = RGCN(args.n_hidden,
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases,
                 dropout=args.dropout,
                 self_loop=args.use_self_loop)

    return embed_layer, model

def gen_norm(g):
    _, v, eid = g.all_edges(form='all')
    _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0], device=eid.device) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm

def train(model, embed_layer, train_loader, inv_target,
          device, labels, emb_optimizer, optimizer):
    model.train()
    embed_layer.train()

    for sample_data in train_loader:
        _, seeds, blocks = sample_data
        # map the seed nodes back to their type-specific ids,
        # in order to get the target node labels
        seeds = inv_target[seeds]
        for blc in blocks:
            gen_norm(blc)

        feats = embed_layer(blocks[0].srcdata[dgl.NID],
                            blocks[0].srcdata['ntype'],
                            blocks[0].srcdata['type_id'])
        blocks = [blc.to(device) for blc in blocks]
        logits = model(blocks, feats)
        loss = F.cross_entropy(logits, labels[seeds])
        emb_optimizer.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        emb_optimizer.step()
        optimizer.step()

        train_acc = th.mean(logits.argmax(dim=1) == labels[seeds]).item()

    return train_acc, loss

def evaluate(device, model, embed_layer, eval_loader, inv_target):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in tqdm(eval_loader):
            _, seeds, blocks = sample_data
            seeds = inv_target[seeds]

            for blc in blocks:
                gen_norm(blc)

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'])
            blocks = [blc.to(device) for blc in blocks]
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    return eval_logits, eval_seeds

def main(args):
    hg, g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target = load_data(
        args.dataset, inv_target=True)
    node_feats = [hg.num_nodes(ntype) for ntype in hg.ntypes]

    # Create csr/coo/csc formats before launching training processes.
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()

    device = th.device(args.gpu if args.gpu >= 0 else 'cpu')
    train_loader, val_loader, test_loader = init_dataloaders(
        args, g, train_idx, test_idx, target_idx, args.gpu)
    embed_layer, model = init_models(args, device, node_feats, num_classes, num_rels)

    if args.gpu >= 0:
        th.cuda.set_device(device)
        labels = labels.to(device)
        model = model.to(device)
        # with dgl_sparse emb, only node embedding is not on GPU
        if args.dgl_sparse:
            embed_layer.cuda(args.gpu)

    # optimizer
    if args.dgl_sparse:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=1e-2, weight_decay=args.l2norm)
        emb_optimizer = dgl.optim.SparseAdam(params=embed_layer.dgl_emb, lr=args.sparse_lr, eps=1e-8)
    else:
        dense_params = list(model.parameters())
        optimizer = th.optim.Adam(dense_params, lr=1e-2, weight_decay=args.l2norm)
        embs = list(embed_layer.node_embeds.parameters())
        emb_optimizer = th.optim.SparseAdam(embs, lr=args.sparse_lr)

    for epoch in range(args.n_epochs):
        train_acc, loss = train(model, embed_layer, train_loader, inv_target, device,
                                labels, emb_optimizer, optimizer)
        print("Epoch {:05d}/{:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, args.n_epochs, train_acc, loss.item()))

        gc.collect()

        val_logits, val_seeds = evaluate(device, model, embed_layer, val_loader, inv_target)
        val_loss, val_acc = F.cross_entropy(val_logits, labels[val_seeds].cpu()).item(), \
            th.mean(val_logits.argmax(dim=1) == labels[val_seeds].cpu()).item()
        print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(val_acc, val_loss))

    test_logits, test_seeds = evaluate(device, model, embed_layer,
                                       test_loader, inv_target)
    test_loss, test_acc = F.cross_entropy(test_logits, labels[test_seeds].cpu()).item(), \
        th.mean(test_logits.argmax(dim=1) == labels[test_seeds].cpu()).item()
    print("Final Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification with sampling')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--sparse-lr", type=float, default=2e-2,
                        help="sparse embedding learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=5e-4,
                        help="l2 norm coef")
    parser.add_argument("--fanout", type=str, default="4, 4",
                        help="Fan-out of neighbor sampling")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Mini-batch size")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
                        help='Use sparse embedding for node embeddings.')
    args = parser.parse_args()

    print(args)
    main(args)
