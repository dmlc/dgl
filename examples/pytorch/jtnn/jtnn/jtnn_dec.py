import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab
from .nnutils import GRUUpdate, cuda
import copy
import itertools
from dgl import batch, dfs_labeled_edges_generator
import dgl.function as DGLF
import networkx as nx
from .line_profiler_integration import profile
import numpy as np

MAX_NB = 8
MAX_DECODE_LEN = 100

'''
def _dfs(trace, forest, i, cur, parent):
    _, next_, down_eid = forest.out_edges(cur, 'all')
    prev, _, up_eid = forest.in_edges(cur, 'all')
    down_eid = dict(zip(next_.tolist(), down_eid.tolist()))
    up_eid = dict(zip(prev.tolist(), up_eid.tolist()))
    for n in down_eid:
        if n == parent:
            continue
        trace.append((cur, n, down_eid[n], i, 1))
        _dfs(trace, forest, i, n, cur)
        trace.append((n, cur, up_eid[n], i, 0))

def dfs_order(forest, roots):
    edge_list = []
    
    for i, root in enumerate(roots):
        trace = []
        _dfs(trace, forest, i, root, None)
        edge_list.append(trace)

    for edges in itertools.zip_longest(*edge_list):
        edges = (e for e in edges if e is not None)
        u, v, e, i, p = zip(*edges)
        yield u, v, e, i, p
'''

def dfs_order(forest, roots):
    edges = dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
    for e, l in zip(*edges):
        # I exploited the fact that the reverse edge ID equal to 1 xor forward
        # edge ID for molecule trees.  Normally, I should locate reverse edges
        # using find_edges().
        yield e ^ l, l

dec_tree_node_msg = DGLF.copy_edge(edge='m', out='m')
dec_tree_node_reduce = DGLF.sum(msg='m', out='h')


def dec_tree_node_update(nodes):
    return {'new': nodes.data['new'].clone().zero_()}


dec_tree_edge_msg = [DGLF.copy_src(src='m', out='m'), DGLF.copy_src(src='rm', out='rm')]
dec_tree_edge_reduce = [DGLF.sum(msg='m', out='s'), DGLF.sum(msg='rm', out='accum_rm')]


class DGLJTNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.dec_tree_edge_update = GRUUpdate(hidden_size)

        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

    def forward(self, mol_trees, tree_vec):
        '''
        The training procedure which computes the prediction loss given the
        ground truth tree
        '''
        mol_tree_batch = batch(mol_trees)
        mol_tree_batch_lg = mol_tree_batch.line_graph(backtracking=False, shared=True)
        n_trees = len(mol_trees)

        return self.run(mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec)

    def run(self, mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec):
        node_offset = np.cumsum([0] + mol_tree_batch.batch_num_nodes)
        root_ids = node_offset[:-1]
        n_nodes = mol_tree_batch.number_of_nodes()
        n_edges = mol_tree_batch.number_of_edges()

        mol_tree_batch.ndata.update({
            'x': self.embedding(mol_tree_batch.get_n_repr()['wid']),
            'h': cuda(torch.zeros(n_nodes, self.hidden_size)),
            'new': cuda(torch.ones(n_nodes).byte()),  # whether it's newly generated node
        })

        mol_tree_batch.edata.update({
            's': cuda(torch.zeros(n_edges, self.hidden_size)),
            'm': cuda(torch.zeros(n_edges, self.hidden_size)),
            'r': cuda(torch.zeros(n_edges, self.hidden_size)),
            'z': cuda(torch.zeros(n_edges, self.hidden_size)),
            'src_x': cuda(torch.zeros(n_edges, self.hidden_size)),
            'dst_x': cuda(torch.zeros(n_edges, self.hidden_size)),
            'rm': cuda(torch.zeros(n_edges, self.hidden_size)),
            'accum_rm': cuda(torch.zeros(n_edges, self.hidden_size)),
        })

        mol_tree_batch.apply_edges(
            func=lambda edges: {'src_x': edges.src['x'], 'dst_x': edges.dst['x']},
        )

        # input tensors for stop prediction (p) and label prediction (q)
        p_inputs = []
        p_targets = []
        q_inputs = []
        q_targets = []

        # Predict root
        mol_tree_batch.pull(
            root_ids,
            dec_tree_node_msg,
            dec_tree_node_reduce,
            dec_tree_node_update,
        )
        # Extract hidden states and store them for stop/label prediction
        h = mol_tree_batch.nodes[root_ids].data['h']
        x = mol_tree_batch.nodes[root_ids].data['x']
        p_inputs.append(torch.cat([x, h, tree_vec], 1))
        # If the out degree is 0 we don't generate any edges at all
        t_set = list(range(len(root_ids)))
        root_out_degrees = mol_tree_batch.out_degrees(root_ids)
        t_finalize = [i for i in range(len(root_ids)) if root_out_degrees[i] == 0]
        q_inputs.append(torch.cat([h, tree_vec], 1))
        q_targets.append(mol_tree_batch.nodes[root_ids].data['wid'])

        # Traverse the tree and predict on children
        for eid, p in dfs_order(mol_tree_batch, root_ids):
            u, v = mol_tree_batch.find_edges(eid)

            p_target_list = []
            _i = 0
            for i in t_set:
                if i in t_finalize:
                    p_target = 0
                else:
                    p_target = 1 - p[_i]
                    _i += 1
                p_target_list.append(p_target)
            p_targets.append(torch.tensor(p_target_list))
            t_set = [i for i in t_set if i not in t_finalize]
            t_finalize = [t_set[i] for i in range(len(t_set)) if np.isin(v, root_ids)[i]]

            mol_tree_batch_lg.pull(
                eid,
                dec_tree_edge_msg,
                dec_tree_edge_reduce,
                self.dec_tree_edge_update,
            )
            is_new = mol_tree_batch.nodes[v].data['new']
            mol_tree_batch.pull(
                v,
                dec_tree_node_msg,
                dec_tree_node_reduce,
                dec_tree_node_update,
            )
            # Extract
            n_repr = mol_tree_batch.nodes[v].data
            h = n_repr['h']
            x = n_repr['x']
            tree_vec_set = tree_vec[t_set]
            wid = n_repr['wid']
            p_inputs.append(torch.cat([x, h, tree_vec_set], 1))
            # Only newly generated nodes are needed for label prediction
            # NOTE: The following works since the uncomputed messages are zeros.
            q_inputs.append(torch.cat([h, tree_vec_set], 1)[is_new])
            q_targets.append(wid[is_new])
        p_targets.append(torch.tensor([0] * len(t_finalize)))

        # Batch compute the stop/label prediction losses
        p_inputs = torch.cat(p_inputs, 0)
        p_targets = cuda(torch.cat(p_targets, 0))
        q_inputs = torch.cat(q_inputs, 0)
        q_targets = torch.cat(q_targets, 0)

        q = self.W_o(torch.relu(self.W(q_inputs)))
        p = self.U_s(torch.relu(self.U(p_inputs)))[:, 0]

        p_loss = F.binary_cross_entropy_with_logits(
            p, p_targets.float(), size_average=False
        ) / n_trees
        q_loss = F.cross_entropy(q, q_targets, size_average=False) / n_trees
        p_acc = ((p > 0).long() == p_targets).sum().float() / p_targets.shape[0]
        q_acc = (q.max(1)[1] == q_targets).float().sum() / q_targets.shape[0]

        return q_loss, p_loss, q_acc, p_acc
