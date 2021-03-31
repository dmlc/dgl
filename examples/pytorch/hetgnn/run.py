import dgl
import time
import tqdm
import torch as th
import numpy as np
import torch.nn as nn
import argparse
from config import Config
from torch.utils.data import IterableDataset, DataLoader
from sampler import SkipGramBatchSampler, HetGNNCollator, NeighborSampler
from model import HetGNN
from utils import load_HIN, trans_feature, compute_loss, extract_feature, load_link_pred, author_link_prediction, hetgnn_graph, Hetgnn_evaluate


def run_HetGNN(model, hg, het_graph, config):
    # het_graph is used to sample neighbour
    hg = hg.to('cpu')
    category = config.category
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('label')
    emd =hg.nodes[category].data['dw_embedding']
    train_batch = load_link_pred('./a_a_list_train.txt')
    test_batch = load_link_pred('./a_a_list_test.txt')
    # HetGNN Sampler
    batch_sampler = SkipGramBatchSampler(hg, config.batch_size, config.window_size)
    neighbor_sampler = NeighborSampler(het_graph, hg.ntypes, batch_sampler.num_nodes, config.device)
    collator = HetGNNCollator(neighbor_sampler, hg)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=config.num_workers)

    opt = th.optim.Adam(model.parameters())

    pred = ScorePredictor()
    dataloader_it = iter(dataloader)
    for i in range(config.max_epoch):
        model.train()
        for batch_id in tqdm.trange(config.batches_per_epoch):
            positive_graph, negative_graph, blocks = next(dataloader_it)
            blocks = [b.to(config.device) for b in blocks]
            positive_graph = positive_graph.to(config.device)
            negative_graph = negative_graph.to(config.device)
            # we need extract multi-feature
            input_features = extract_feature(blocks[0], hg.ntypes)

            x = model(blocks[0], input_features)
            loss = compute_loss(pred(positive_graph, x), pred(negative_graph, x))

            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch {:05d} |Train - Loss: {:.4f}'.format(i, loss.item()))
        input_features = extract_feature(het_graph, hg.ntypes)
        x = model(het_graph, input_features)
        author_link_prediction(x['author'].to('cpu').detach(), train_batch, test_batch)
        micro_f1, macro_f1 = Hetgnn_evaluate(x[config.category].to('cpu').detach(), labels, train_idx, test_idx)
        print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
    pass


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

def main(config):
    hg, category, num_classes = load_HIN(config.dataset)
    config.category = category
    config.num_classes = num_classes
    hg = hg.to(config.device)

    hetg = hetgnn_graph(hg, config.dataset)
    het_graph = hetg.get_hetgnn_graph(config.rw_length, config.rw_walks, config.rwr_prob).to('cpu')
    hg = hg.to('cpu')
    het_graph = trans_feature(hg, het_graph)
    model = HetGNN(hg.ntypes, config.dim).to(config.device)
    run_HetGNN(model, hg, het_graph, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='HetGNN', type=str, help='name of models')
    parser.add_argument('--dataset', '-d', default='academic', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')

    args = parser.parse_args()

    config_file = ["./config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, gpu=args.gpu)
    main(config=config)