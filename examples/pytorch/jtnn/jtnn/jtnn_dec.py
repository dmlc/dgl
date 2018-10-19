import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab
from .nnutils import GRUUpdate, cuda
import copy
import itertools
from dgl import batch
import dgl.function as DGLF
import networkx as nx
from .line_profiler_integration import profile

MAX_NB = 8
MAX_DECODE_LEN = 100


def dfs_order(forest, roots):
    '''
    Returns edge source, edge destination, tree ID, and whether u is generating
    a new children
    '''
    forest = forest.to_networkx()
    edge_list = []

    for i, root in enumerate(roots):
        edge_list.append([])
        # The following gives the DFS order on edge on a tree.
        for u, v, t in nx.dfs_labeled_edges(forest, root):
            if u == v or t == 'nontree':
                continue
            elif t == 'forward':
                edge_list[-1].append((u, v, i, 1))
            elif t == 'reverse':
                edge_list[-1].append((v, u, i, 0))

    for edges in itertools.zip_longest(*edge_list):
        edges = (e for e in edges if e is not None)
        u, v, i, p = zip(*edges)
        yield u, v, i, p


dec_tree_node_msg = DGLF.copy_edge(edge='m', out='m')
dec_tree_node_reduce = DGLF.sum(msg='m', out='h')


def dec_tree_node_update(node):
    return {'new': node['new'].clone().zero_()}


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

    @profile
    def forward(self, mol_trees, tree_vec):
        '''
        The training procedure which computes the prediction loss given the
        ground truth tree
        '''
        mol_tree_batch = batch(mol_trees)
        mol_tree_batch_lg = mol_tree_batch.line_graph(backtracking=False, shared=True)
        n_trees = len(mol_trees)

        return self.run(mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec)

    @profile
    def run(self, mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec):
        root_ids = mol_tree_batch.node_offset[:-1]
        n_nodes = len(mol_tree_batch.nodes)
        edge_list = mol_tree_batch.edge_list
        n_edges = len(edge_list)

        mol_tree_batch.set_n_repr({
            'x': self.embedding(mol_tree_batch.get_n_repr()['wid']),
            'h': cuda(torch.zeros(n_nodes, self.hidden_size)),
            'new': cuda(torch.ones(n_nodes).byte()),  # whether it's newly generated node
        })

        mol_tree_batch.set_e_repr({
            's': cuda(torch.zeros(n_edges, self.hidden_size)),
            'm': cuda(torch.zeros(n_edges, self.hidden_size)),
            'r': cuda(torch.zeros(n_edges, self.hidden_size)),
            'z': cuda(torch.zeros(n_edges, self.hidden_size)),
            'src_x': cuda(torch.zeros(n_edges, self.hidden_size)),
            'dst_x': cuda(torch.zeros(n_edges, self.hidden_size)),
            'rm': cuda(torch.zeros(n_edges, self.hidden_size)),
            'accum_rm': cuda(torch.zeros(n_edges, self.hidden_size)),
        })

        mol_tree_batch.update_edge(
            #*zip(*edge_list),
            edge_func=lambda src, dst, edge: {'src_x': src['x'], 'dst_x': dst['x']},
            batchable=True,
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
            batchable=True,
        )
        # Extract hidden states and store them for stop/label prediction
        h = mol_tree_batch.get_n_repr(root_ids)['h']
        x = mol_tree_batch.get_n_repr(root_ids)['x']
        p_inputs.append(torch.cat([x, h, tree_vec], 1))
        t_set = list(range(len(root_ids)))
        q_inputs.append(torch.cat([h, tree_vec], 1))
        q_targets.append(mol_tree_batch.get_n_repr(root_ids)['wid'])

        # Traverse the tree and predict on children
        for u, v, i, p in dfs_order(mol_tree_batch, root_ids):
            assert set(t_set).issuperset(i)
            ip = dict(zip(i, p))
            # TODO: context
            p_targets.append(cuda(torch.tensor([ip.get(_i, 0) for _i in t_set])))
            t_set = list(i)
            eid = mol_tree_batch.get_edge_id(u, v)
            mol_tree_batch_lg.pull(
                eid,
                dec_tree_edge_msg,
                dec_tree_edge_reduce,
                self.dec_tree_edge_update,
                batchable=True,
            )
            is_new = mol_tree_batch.get_n_repr(v)['new']
            mol_tree_batch.pull(
                v,
                dec_tree_node_msg,
                dec_tree_node_reduce,
                dec_tree_node_update,
                batchable=True,
            )
            # Extract
            h = mol_tree_batch.get_n_repr(v)['h']
            x = mol_tree_batch.get_n_repr(v)['x']
            p_inputs.append(torch.cat([x, h, tree_vec[t_set]], 1))
            # Only newly generated nodes are needed for label prediction
            # NOTE: The following works since the uncomputed messages are zeros.
            q_inputs.append(torch.cat([h[is_new], tree_vec[t_set][is_new]], 1))
            q_targets.append(mol_tree_batch.get_n_repr(v)['wid'][is_new])
        p_targets.append(cuda(torch.tensor([0 for _ in t_set])))

        # Batch compute the stop/label prediction losses
        p_inputs = torch.cat(p_inputs, 0)
        p_targets = torch.cat(p_targets, 0)
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
