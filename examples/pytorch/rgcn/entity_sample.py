"""
Differences compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import torch as th
import torch.nn.functional as F
import dgl

from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm

from entity_utils import load_data
from model import RelGraphEmbedLayer, RGCN

def init_dataloaders(args, g, train_idx, test_idx, target_idx, device, use_ddp=False):
    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    sampler = MultiLayerNeighborSampler(fanouts)

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

    # -1 for sampling all neighbors
    test_sampler = MultiLayerNeighborSampler([-1] * len(fanouts))
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

def init_models(args, device, num_nodes, num_classes, num_rels):
    embed_layer = RelGraphEmbedLayer(device,
                                     num_nodes,
                                     args.n_hidden)

    model = RGCN(args.n_hidden,
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases,
                 dropout=args.dropout,
                 self_loop=args.use_self_loop)

    return embed_layer, model

def process_batch(inv_target, batch):
    _, seeds, blocks = batch
    # map the seed nodes back to their type-specific ids,
    # in order to get the target node labels
    seeds = inv_target[seeds]

    for blc in blocks:
        blc.edata['norm'] = dgl.norm_by_dst(blc).unsqueeze(1)

    return seeds, blocks

def train(model, embed_layer, train_loader, inv_target,
          labels, emb_optimizer, optimizer):
    model.train()
    embed_layer.train()

    for sample_data in train_loader:
        seeds, blocks = process_batch(inv_target, sample_data)
        feats = embed_layer(blocks[0].srcdata[dgl.NID].cpu())
        logits = model(blocks, feats)
        loss = F.cross_entropy(logits, labels[seeds])
        emb_optimizer.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        emb_optimizer.step()
        optimizer.step()

        train_acc = accuracy(logits.argmax(dim=1), labels[seeds]).item()

    return train_acc, loss.item()

def evaluate(model, embed_layer, eval_loader, inv_target):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        for sample_data in tqdm(eval_loader):
            seeds, blocks = process_batch(inv_target, sample_data)
            feats = embed_layer(blocks[0].srcdata[dgl.NID].cpu())
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    return eval_logits, eval_seeds

def main(args):
    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target = load_data(
        args.dataset, inv_target=True)

    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')

    train_loader, val_loader, test_loader = init_dataloaders(
        args, g, train_idx, test_idx, target_idx, args.gpu)
    embed_layer, model = init_models(args, device, g.num_nodes(), num_classes, num_rels)

    labels = labels.to(device)
    model = model.to(device)

    emb_optimizer = th.optim.SparseAdam(embed_layer.parameters(), lr=args.sparse_lr)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=args.l2norm)

    for epoch in range(args.n_epochs):
        train_acc, loss = train(model, embed_layer, train_loader, inv_target,
                                labels, emb_optimizer, optimizer)
        print("Epoch {:05d}/{:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, args.n_epochs, train_acc, loss))

        val_logits, val_seeds = evaluate(model, embed_layer, val_loader, inv_target)
        val_acc = accuracy(val_logits.argmax(dim=1), labels[val_seeds].cpu()).item()
        print("Validation Accuracy: {:.4f}".format(val_acc))

    test_logits, test_seeds = evaluate(model, embed_layer,
                                       test_loader, inv_target)
    test_acc = accuracy(test_logits.argmax(dim=1), labels[test_seeds].cpu()).item()
    print("Final Test Accuracy: {:.4f}".format(test_acc))

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
    args = parser.parse_args()

    print(args)
    main(args)
