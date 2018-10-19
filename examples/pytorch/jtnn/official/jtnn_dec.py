import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree, MolTreeNode
from .nnutils import create_var, GRU, GRUUpdate, cuda
from .chemutils import enum_assemble
import copy
import itertools
from dgl import batch
import dgl.function as DGLF
import networkx as nx
from .line_profiler_integration import profile

MAX_NB = 8
MAX_DECODE_LEN = 100

class JTNNDecoder(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        #Feature Aggregate Weights
        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)

        #Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.idx = -1
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x,y,z in trace]
       
    @profile
    def forward(self, mol_batch, mol_vec):
        super_root = MolTreeNode("")
        super_root.idx = -1

        #Initialize
        pred_hiddens,pred_mol_vecs,pred_targets = [],[],[]
        stop_hiddens,stop_targets = [],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], super_root)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        #Predict Root
        pred_hiddens.append(create_var(cuda(torch.zeros(len(mol_batch),self.hidden_size))))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_mol_vecs.append(mol_vec) 

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(cuda(torch.zeros(self.hidden_size)), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]

            for node_x,real_y,_ in prop_list:
                #Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                #Current clique embedding
                cur_x.append(node_x.wid)

            #Clique embedding
            cur_x = create_var(cuda(torch.LongTensor(cur_x)))
            cur_x = self.embedding(cur_x)

            #Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            #Hidden states for stop prediction
            cur_batch = create_var(cuda(torch.LongTensor(batch_list)))
            cur_mol_vec = mol_vec.index_select(0, cur_batch)
            stop_hidden = torch.cat([cur_x,cur_o,cur_mol_vec], dim=1)
            stop_hiddens.append( stop_hidden )
            stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(cuda(torch.LongTensor(batch_list)))
                pred_mol_vecs.append( mol_vec.index_select(0, cur_batch) )

                cur_pred = create_var(cuda(torch.LongTensor(pred_list)))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(cuda(torch.LongTensor(cur_x)))
        cur_x = self.embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x,cur_o,mol_vec], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_mol_vecs = torch.cat(pred_mol_vecs, dim=0)
        pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_targets = create_var(cuda(torch.LongTensor(pred_targets)))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_vecs = nn.ReLU()(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()
        stop_targets = create_var(cuda(torch.Tensor(stop_targets)))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.data[0], stop_acc.data[0]
    
    def decode(self, mol_vec, prob_decode):
        stack,trace = [],[]
        init_hidden = create_var(torch.zeros(1,self.hidden_size))
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))

        #Root Prediction
        root_hidden = torch.cat([init_hidden, mol_vec], dim=1)
        root_hidden = nn.ReLU()(self.W(root_hidden))
        root_score = self.W_o(root_hidden)
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.data[0]

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, self.vocab.get_slots(root.wid)) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x,fa_slot = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(cuda(torch.LongTensor([node_x.wid])))
            cur_x = self.embedding(cur_x)

            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hidden = torch.cat([cur_x,cur_h,mol_vec], dim=1)
            stop_hidden = nn.ReLU()(self.U(stop_hidden))
            stop_score = nn.Sigmoid()(self.U_s(stop_hidden) * 20).squeeze()
            
            if prob_decode:
                backtrack = (torch.bernoulli(1.0 - stop_score.data)[0] == 1)
            else:
                backtrack = (stop_score.data[0] < 0.5)

            if not backtrack: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_hidden = torch.cat([new_h,mol_vec], dim=1)
                pred_hidden = nn.ReLU()(self.W(pred_hidden))
                pred_score = nn.Softmax()(self.W_o(pred_hidden) * 20)
                if prob_decode:
                    sort_wid = torch.multinomial(pred_score.data.squeeze(), 5)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = step + 1
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            if backtrack: #Backtrack, use if instead of else
                if len(stack) == 1: 
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

"""
Helper Functions:
"""

def dfs(stack, x, fa):
    for y in x.neighbors:
        if y.idx == fa.idx:
            continue
        stack.append((x,y,1))
        dfs(stack, y, x)
        stack.append((y,x,0))

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = list(zip(*matches))
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True
    
def can_assemble(node_x, node_y):
    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0


def dfs_order(forest, roots):
    '''
    Returns edge source, edge destination, tree ID, and whether u is generating
    a new children
    '''
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
#def dec_tree_node_msg(src, edge):
#    return edge['m']


dec_tree_node_reduce = DGLF.sum(msg='m', out='h')
#def dec_tree_node_reduce(node, msgs):
#    return {'h': msgs.sum(1)}


def dec_tree_node_update(node):
    return {'new': node['new'].clone().zero_()}


dec_tree_edge_msg = [DGLF.copy_src(src='m', out='m'), DGLF.copy_src(src='rm', out='rm')]
#def dec_tree_edge_msg(src, edge):
#    return {'m': src['m'], 'rm': src['r'] * src['m']}


dec_tree_edge_reduce = [DGLF.sum(msg='m', out='s'), DGLF.sum(msg='rm', out='accum_rm')]
#def dec_tree_edge_reduce(node, msgs):
#    return {'s': msgs['m'].sum(1), 'accum_rm': msgs['rm'].sum(1)}


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
        mol_tree_batch_lg = line_graph(mol_tree_batch, no_backtracking=True)
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

    def decode(self, mol_vec, prob_decode):
        # Using non-batched DGL to decode since it involves simultaneous graph
        # generation and message passing
        pass
