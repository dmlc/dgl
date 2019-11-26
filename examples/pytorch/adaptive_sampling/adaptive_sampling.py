import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as ssp
from dgl.data import citation_graph as citegrh
import networkx as nx

##load data
##cora dataset have 2708 nodes, 1208 of them is used as train set 1000 of them is used as test set
data = citegrh.load_cora()
adj = nx.adjacency_matrix(data.graph)
# reorder
n_nodes = 2708
ids_shuffle = np.arange(n_nodes)
np.random.shuffle(ids_shuffle)
adj = adj[ids_shuffle, :][:, ids_shuffle]
data.features = data.features[ids_shuffle]
data.labels = data.labels[ids_shuffle]
##train-test split
train_nodes = np.arange(1208)
test_nodes = np.arange(1708, 2708)
train_adj = adj[train_nodes, :][:, train_nodes]
test_adj = adj[test_nodes, :][:, test_nodes]
trainG = dgl.DGLGraph(train_adj)
allG = dgl.DGLGraph(adj)
h = torch.tensor(data.features[train_nodes], dtype=torch.float32)
test_h = torch.tensor(data.features[test_nodes], dtype=torch.float32)
all_h = torch.tensor(data.features, dtype=torch.float32)
train_nodes = torch.tensor(train_nodes)
test_nodes = torch.tensor(test_nodes)
y_train = torch.tensor(data.labels[train_nodes])
y_test = torch.tensor(data.labels[test_nodes])
input_size = h.shape[1]
output_size = data.num_labels

##configuration
config = {
    'n_epoch': 300,
    'lamb': 0.5,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'hidden_size': 16,
    ##sample size for each layer during training
    'batch_size': 256,
    ##sample size for each layer during test
    'test_batch_size': 64,
    'test_layer_sizes': [64 * 8, 64 * 8],
}


class NodeSampler(object):
    """Minibatch sampler that samples batches of nodes uniformly from the given graph and list of seeds.
    """

    def __init__(self, graph, seeds, batch_size):
        self.seeds = seeds
        self.batch_size = batch_size
        self.graph = graph

    def __len__(self):
        return len(self.seeds) // self.batch_size

    def __iter__(self):
        """Returns
        (1) The seed node IDs, for NodeFlow generation,
        (2) Indices of the seeds in the original seed array, as auxiliary data.
        """
        batches = torch.randperm(len(self.seeds)).split(self.batch_size)
        for i in range(len(self)):
            if len(batches[i]) < self.batch_size:
                break
            else:
                yield self.seeds[batches[i]], batches[i]


def create_nodeflow(layer_mappings, block_mappings, block_aux_data, rel_graphs, seed_map):
    hg = dgl.hetero_from_relations(rel_graphs)
    hg.layer_mappings = layer_mappings
    hg.block_mappings = block_mappings
    hg.block_aux_data = block_aux_data
    hg.seed_map = seed_map
    return hg


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = ssp.eye(adj.shape[0]) + adj
    adj = ssp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class AdaptGenerator(object):
    """
    Nodeflow generator used for adaptive sampling

    """

    def __init__(self, graph, num_blocks, node_feature=None, sampler=None, num_workers=0, coalesce=False,
                 sampler_weights=None, layer_nodes=None):
        self.node_feature = node_feature
        adj = graph.adjacency_matrix_scipy()
        adj.data = np.ones(adj.nnz)
        self.norm_adj = normalize_adj(adj).tocsr()
        self.layer_nodes = layer_nodes
        if sampler_weights is not None:
            self.sample_weights = sampler_weights
        else:
            self.sample_weights = nn.Parameter(torch.randn((input_size, 2), dtype=torch.float32))
            nn.init.xavier_uniform_(self.sample_weights)
        self.graph = graph
        self.num_blocks = num_blocks
        self.coalesce = coalesce
        self.sampler = sampler

    def __iter__(self):
        for sampled in self.sampler:
            seeds = sampled[0]
            auxiliary = sampled[1:]
            hg = self(seeds, *auxiliary)
            yield (hg, auxiliary[0])

    def __call__(self, seeds, *auxiliary):
        """
           The __call__ function must take in an array of seeds, and any auxiliary data, and
           return a NodeFlow grown from the seeds and conditioned on the auxiliary data.
        """
        curr_frontier = seeds  # Current frontier to grow neighbors from
        layer_mappings = []  # Mapping from layer node ID to parent node ID
        block_mappings = []  # Mapping from block edge ID to parent edge ID, or -1 if nonexistent
        block_aux_data = []
        rel_graphs = []

        if self.coalesce:
            curr_frontier = torch.LongTensor(np.unique(seeds.numpy()))
            invmap = {x: i for i, x in enumerate(curr_frontier.numpy())}
            seed_map = [invmap[x] for x in seeds.numpy()]
        else:
            seed_map = list(range(len(seeds)))

        layer_mappings.append(curr_frontier.numpy())

        for i in reversed(range(self.num_blocks)):
            neighbor_nodes, neighbor_edges, num_neighbors, aux_result = self.stepback(curr_frontier, i, *auxiliary)
            prev_frontier_srcs = neighbor_nodes
            # The un-coalesced mapping from block edge ID to parent edge ID
            prev_frontier_edges = neighbor_edges.numpy()
            nodes_idx_map = dict({*zip(neighbor_nodes.numpy(), range(len(aux_result)))})
            # Coalesce nodes
            if self.coalesce:
                prev_frontier = np.unique(prev_frontier_srcs.numpy())
                prev_frontier_invmap = {x: j for j, x in enumerate(prev_frontier)}
                block_srcs = np.array([prev_frontier_invmap[s] for s in prev_frontier_srcs.numpy()])
            else:
                prev_frontier = prev_frontier_srcs.numpy()
                block_srcs = np.arange(len(prev_frontier_edges))
            aux_result = aux_result[[nodes_idx_map[i] for i in prev_frontier]]
            block_dsts = np.arange(len(curr_frontier)).repeat(num_neighbors)

            rel_graphs.insert(0, dgl.bipartite(
                (block_srcs, block_dsts),
                'layer%d' % i, 'block%d' % i, 'layer%d' % (i + 1),
                (len(prev_frontier), len(curr_frontier))
            ))

            layer_mappings.insert(0, prev_frontier)
            block_mappings.insert(0, prev_frontier_edges)
            block_aux_data.insert(0, aux_result)

            curr_frontier = torch.LongTensor(prev_frontier)

        return create_nodeflow(
            layer_mappings=layer_mappings,
            block_mappings=block_mappings,
            block_aux_data=block_aux_data,
            rel_graphs=rel_graphs,
            seed_map=seed_map)

    def stepback(self, curr_frontier, layer_index, *auxiliary):
        """Function that takes in the node set in the current layer, and returns the
        neighbors of each node.

        Parameters
        ----------
        curr_frontier : Tensor
        auxiliary : any auxiliary data yielded by the sampler

        Returns
        -------
        neighbor_nodes, incoming_edges, num_neighbors, auxiliary: Tensor, Tensor, Tensor, any
            num_neighbors[i] contains the number of neighbors generated for curr_frontier[i]

            neighbor_nodes[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            neighbors as node IDs in the original graph for curr_frontier[i].

            incoming_edges[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            incoming edges as edge IDs in the original graph for curr_frontier[i], or -1 if the
            edge does not exist, or if we don't care about the edge, in the original graph.

            auxiliary could be of any type containing block-specific additional data.
        """

        # Relies on that the same dst node of in_edges are contiguous, and the dst nodes
        # are ordered the same as curr_frontier.

        sample_weights = self.sample_weights
        layer_size = self.layer_nodes[layer_index]

        src, des, eid = self.graph.in_edges(curr_frontier, form='all')
        neighbor_nodes = torch.unique(torch.cat((curr_frontier, src), dim=0), sorted=False)
        sparse_adj = self.norm_adj[curr_frontier, :][:, neighbor_nodes]
        square_adj = sparse_adj.multiply(sparse_adj).sum(0)

        tensor_adj = torch.FloatTensor(square_adj.A[0])
        ##compute sampling probability for next layer which is decided by :
        # 1. attention part derived from node hidden feature
        # 2. adjacency part derived from graph structure
        hu = torch.matmul(self.node_feature[neighbor_nodes], sample_weights[:, 0])
        hv = torch.sum(torch.matmul(self.node_feature[curr_frontier], sample_weights[:, 1]))
        adj_part = torch.sqrt(tensor_adj)
        attention_part = F.relu(hv + hu) + 1
        gu = F.relu(hu) + 1
        probas = adj_part * attention_part * gu
        probas = probas / torch.sum(probas)
        ##build graph between candidates and curr_frontier
        candidates = neighbor_nodes[probas.multinomial(num_samples=layer_size, replacement=True)]
        ivmap = {x: i for i, x in enumerate(neighbor_nodes.numpy())}
        # use matrix operation in pytorch to avoid for-loop
        curr_padding = curr_frontier.repeat_interleave(len(candidates))
        cand_padding = candidates.repeat(len(curr_frontier))
        ##the edges between candidates and curr_frontier composed of
        # 1. edges in orginal graph
        # 2. edges between same node(self-loop)
        has_loops = curr_padding == cand_padding
        has_edges = self.graph.has_edges_between(cand_padding, curr_padding)
        loops_or_edges = (has_edges.bool() + has_loops).int()
        # get neighbor_nodes and corresponding sampling probability
        num_neighbors = loops_or_edges.reshape((len(curr_frontier), -1)).sum(1)
        sample_neighbor = cand_padding[loops_or_edges.bool()]
        q_prob = probas[[ivmap[i] for i in sample_neighbor.numpy()]]

        # get the edge mapping ,-1 if the edge dostn't exist
        eids = torch.zeros(torch.sum(num_neighbors), dtype=torch.int64) - 1
        has_edge_ids = torch.where(has_edges)[0]
        all_ids = torch.where(loops_or_edges)[0]
        edges_ids_map = torch.where(has_edge_ids[:, None] == all_ids[None, :])[1]
        eids[edges_ids_map] = self.graph.edge_ids(cand_padding, curr_padding)

        return sample_neighbor, eids, num_neighbors, q_prob


class AdaptSAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 G=None):
        super(AdaptSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # self.fc_self = nn.Linear(in_feats, out_feats, bias=bias).double()
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()
        self.G = G

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, hidden_feat, node_feat, layer_id, sample_weights, norm_adj=None, var_loss=None,
                is_test=False):
        """
        graph: Bipartite.  Has two edge types.  The first one represents the connection to
        the desired nodes from neighbors.  The second one represents the computation
        dependence of the desired nodes themselves.
        :type graph: dgl.DGLHeteroGraph
        """
        # local_var is not implemented for heterograph
        # graph = graph.local_var()

        neighbor_etype_name = 'block%d' % layer_id
        src_name = 'layer%d' % layer_id
        dst_name = 'layer%d' % (layer_id + 1)

        graph.nodes[src_name].data['hidden_feat'] = hidden_feat
        graph.nodes[src_name].data['node_feat'] = node_feat[graph.layer_mappings[layer_id]]
        ##normalized degree for node_i is defined as  1/sqrt(d_i+1)
        ##use the training graph during training and whole graph during testing
        if not is_test:
            graph.nodes[src_name].data['norm_deg'] = 1 / torch.sqrt(
                trainG.in_degrees(graph.layer_mappings[layer_id]).float() + 1)
            graph.nodes[dst_name].data['norm_deg'] = 1 / torch.sqrt(
                trainG.in_degrees(graph.layer_mappings[layer_id + 1]).float() + 1)
        else:
            graph.nodes[src_name].data['norm_deg'] = 1 / torch.sqrt(
                allG.in_degrees(graph.layer_mappings[layer_id]).float() + 1)
            graph.nodes[dst_name].data['norm_deg'] = 1 / torch.sqrt(
                allG.in_degrees(graph.layer_mappings[layer_id + 1]).float() + 1)
        graph.nodes[dst_name].data['node_feat'] = node_feat[graph.layer_mappings[layer_id + 1]]
        graph.nodes[src_name].data['q_probs'] = graph.block_aux_data[layer_id]

        def send_func(edges):

            hu = torch.matmul(edges.src['node_feat'], sample_weights[:, 0])
            hv = torch.matmul(edges.dst['node_feat'], sample_weights[:, 1])
            ##attention coeffient is adjusted by normalized degree and sampling probability
            attentions = edges.src['norm_deg'] * edges.dst['norm_deg'] * (F.relu(hu + hv) + 0.1) / edges.src[
                'q_probs'] / len(hu)

            hidden = edges.src['hidden_feat'] * torch.reshape(attentions, [-1, 1])
            return {"hidden": hidden}

        recv_func = dgl.function.sum('hidden', 'neigh')
        # def receive_fuc(nodes):
        # aggregate from neighbors
        graph[neighbor_etype_name].update_all(message_func=send_func, reduce_func=recv_func)
        h_neigh = graph.nodes[dst_name].data['neigh']
        rst = self.fc_neigh(h_neigh)

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        # compute the variance loss according to the formula in orginal paper
        if var_loss and not is_test:
            pre_sup = self.fc_neigh(hidden_feat)  # u*h
            ##normalized adjacency matrix for nodeflow layer
            support = norm_adj[graph.layer_mappings[layer_id + 1], :][:, graph.layer_mappings[layer_id]]  ##v*u
            hu = torch.matmul(node_feat[graph.layer_mappings[layer_id]], sample_weights[:, 0])
            hv = torch.matmul(node_feat[graph.layer_mappings[layer_id + 1]], sample_weights[:, 1])
            attentions = (F.relu(torch.reshape(hu, [1, -1]) + torch.reshape(hv, [-1, 1])) + 1) / graph.block_aux_data[
                layer_id] / len(hu)
            adjust_support = torch.tensor(support.A, dtype=torch.float32) * attentions
            support_mean = adjust_support.sum(0)
            mu_v = torch.mean(rst, dim=0)  # h
            diff = torch.reshape(support_mean, [-1, 1]) * pre_sup - torch.reshape(mu_v, [1, -1])
            loss = torch.sum(diff * diff) / len(hu) / len(hv)
            return rst, loss
        return rst


class AdaptGraphSAGENet(nn.Module):
    def __init__(self, sample_weights, node_feature, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaptSAGEConv(input_size, hidden_size, 'mean', bias=False, activation=F.relu),
            AdaptSAGEConv(hidden_size, output_size, 'mean', bias=False, activation=F.relu),
        ])
        self.sample_weights = sample_weights
        self.node_feature = node_feature
        self.norm_adj = normalize_adj(trainG.adjacency_matrix_scipy())

    def forward(self, nf, h, is_test=False):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and not is_test:
                h, loss = layer(nf, h, self.node_feature, i, self.sample_weights, norm_adj=self.norm_adj, var_loss=True,
                                is_test=is_test)
            else:
                h = layer(nf, h, self.node_feature, i, self.sample_weights, is_test=is_test)
        if is_test:
            return h
        return h, loss


def main(args):
    config.update(args)
    config['layer_sizes'] = [int(config['node_per_layer']) for _ in range(2)]
    # Create a sampler for training nodes and testing nodes
    train_sampler = NodeSampler(graph=trainG, seeds=train_nodes, batch_size=config['batch_size'])
    test_sampler = NodeSampler(graph=allG, seeds=test_nodes, batch_size=config['test_batch_size'])
    ##Generator for training
    train_generator = AdaptGenerator(graph=trainG, node_feature=all_h, layer_nodes=config['layer_sizes'],
                                     sampler=train_sampler,
                                     num_blocks=len(config['layer_sizes']), coalesce=True)
    # Generator for testing
    test_sample_generator = AdaptGenerator(graph=allG, node_feature=all_h, sampler=test_sampler,
                                           num_blocks=len(config['test_layer_sizes']),
                                           sampler_weights=train_generator.sample_weights,
                                           layer_nodes=config['test_layer_sizes'],
                                           coalesce=True)
    model = AdaptGraphSAGENet(train_generator.sample_weights, all_h, config['hidden_size'])
    params = list(model.parameters())
    params.append(train_generator.sample_weights)
    opt = torch.optim.Adam(params=params, lr=config['lr'])
    # model.train()
    lamb, weight_decay = config['lamb'], config['weight_decay']
    for epoch in range(config['n_epoch']):
        train_accs = []
        for nf, sample_indices in train_generator:
            seed_map = nf.seed_map
            train_y_hat, varloss = model(nf, h[nf.layer_mappings[0]])
            train_y_hat = train_y_hat[seed_map]
            y_train_batch = y_train[sample_indices]
            y_pred = torch.argmax(train_y_hat, dim=1)
            train_acc = torch.sum(torch.eq(y_pred, y_train_batch)).item() / config['batch_size']
            train_accs.append(train_acc)
            loss = F.cross_entropy(train_y_hat.squeeze(), y_train_batch)
            l2_loss = torch.norm(params[0])
            total_loss = varloss * lamb + loss + l2_loss * weight_decay
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            # print(train_sampler.sample_weight)
        test_accs = []
        for test_nf, test_sample_indices in test_sample_generator:
            seed_map = test_nf.seed_map
            # print(test_sample_indices)
            test_y_hat = model(test_nf, all_h[test_nf.layer_mappings[0]], is_test=True)
            # print("test",test_y_hat)
            test_y_hat = test_y_hat[seed_map]
            y_test_batch = y_test[test_sample_indices]
            y_pred = torch.argmax(test_y_hat, dim=1)
            test_acc = torch.sum(torch.eq(y_pred, y_test_batch)).item() / len(y_pred)
            test_accs.append(test_acc)
        print("eqoch{} train accuracy {}, regloss {}, loss {} ,test accuracy {}".format(epoch, np.mean(train_acc),
                                                                                        varloss.item() * lamb,
                                                                                        total_loss.item(),
                                                                                        np.mean(test_accs)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Adaptive Sampling for CCN')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('-l', '--node_per_layer', type=float, default=256,
                        help='sampling size for each layer')

    args = parser.parse_args().__dict__

    main(args)
