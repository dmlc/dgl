import torch
import torch.nn as nn
from collections import deque
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, GRU, GRUUpdate, cuda
import itertools
import networkx as nx
from dgl import batch, unbatch
import dgl.function as DGLF
from .line_profiler_integration import profile

MAX_NB = 8

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    @profile
    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        padding = create_var(cuda(torch.zeros(self.hidden_size)), False)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x,node_y in prop_list:
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = cuda(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                h[(x,y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs

"""
Helper functions
"""

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(cuda(torch.zeros(hidden_size)), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = cuda(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))


def level_order(forest, roots):
    '''
    Given the forest and the list of root nodes,
    returns iterator of list of edges ordered by depth, first in bottom-up
    and then top-down
    '''
    edge_list = []
    node_depth = {}

    edge_list.append([])

    for root in roots:
        node_depth[root] = 0
        for u, v in nx.bfs_edges(forest, root):
            node_depth[v] = node_depth[u] + 1
            if len(edge_list) == node_depth[u]:
                edge_list.append([])
            edge_list[node_depth[u]].append((u, v))

    for edges in reversed(edge_list):
        u, v = zip(*edges)
        yield v, u
    for edges in edge_list:
        u, v = zip(*edges)
        yield u, v


enc_tree_msg = [DGLF.copy_src(src='m', out='m'), DGLF.copy_src(src='rm', out='rm')]
#def enc_tree_msg(src, edge):
#    return {'m': src['m'], 'rm': src['r'] * src['m']}


enc_tree_reduce = [DGLF.sum(msg='m', out='s'), DGLF.sum(msg='rm', out='accum_rm')]
#def enc_tree_reduce(node, msgs):
#    return {'s': msgs['m'].sum(1), 'accum_rm': msgs['rm'].sum(1)}


enc_tree_gather_msg = DGLF.copy_edge(edge='m', out='m')
#def enc_tree_gather_msg(src, edge):
#    return edge['m']


enc_tree_gather_reduce = DGLF.sum(msg='m', out='m')
#def enc_tree_gather_reduce(node, msgs):
#    return {'m': msgs.sum(1)}


class EncoderGatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, node):
        x = node['x']
        m = node['m']
        return {
            'h': torch.relu(self.W(torch.cat([x, m], 1))),
        }


class DGLJTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.enc_tree_update = GRUUpdate(hidden_size)
        self.enc_tree_gather_update = EncoderGatherUpdate(hidden_size)

    @profile
    def forward(self, mol_trees):
        mol_tree_batch = batch(mol_trees)
        
        # Build line graph to prepare for belief propagation
        mol_tree_batch_lg = line_graph(mol_tree_batch, no_backtracking=True)

        return self.run(mol_tree_batch, mol_tree_batch_lg)

    @profile
    def run(self, mol_tree_batch, mol_tree_batch_lg):
        # Since tree roots are designated to 0.  In the batched graph we can
        # simply find the corresponding node ID by looking at node_offset
        root_ids = mol_tree_batch.node_offset[:-1]
        n_nodes = len(mol_tree_batch.nodes)
        edge_list = mol_tree_batch.edge_list
        n_edges = len(edge_list)

        # Assign structure embeddings to tree nodes
        mol_tree_batch.set_n_repr({
            'x': self.embedding(mol_tree_batch.get_n_repr()['wid']),
            'h': cuda(torch.zeros(n_nodes, self.hidden_size)),
        })

        # Initialize the intermediate variables according to Eq (4)-(8).
        # Also initialize the src_x and dst_x fields.
        # TODO: context?
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

        # Send the source/destination node features to edges
        mol_tree_batch.update_edge(
            #*zip(*edge_list),
            edge_func=lambda src, dst, edge: {'src_x': src['x'], 'dst_x': dst['x']},
            batchable=True,
        )

        # Message passing
        # I exploited the fact that the reduce function is a sum of incoming
        # messages, and the uncomputed messages are zero vectors.  Essentially,
        # we can always compute s_ij as the sum of incoming m_ij, no matter
        # if m_ij is actually computed or not.
        for u, v in level_order(mol_tree_batch, root_ids):
            eid = mol_tree_batch.get_edge_id(u, v)
            mol_tree_batch_lg.pull(
                eid,
                enc_tree_msg,
                enc_tree_reduce,
                self.enc_tree_update,
                batchable=True,
            )

        # Readout
        mol_tree_batch.update_all(
            enc_tree_gather_msg,
            enc_tree_gather_reduce,
            self.enc_tree_gather_update,
            batchable=True,
        )

        root_vecs = mol_tree_batch.get_n_repr(root_ids)['h']

        return mol_tree_batch, root_vecs
