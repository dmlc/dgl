import torch
import torch.nn as nn
from .nnutils import cuda
from .chemutils import get_mol
#from mpn import atom_features, bond_features, ATOM_FDIM, BOND_FDIM
import rdkit.Chem as Chem
from dgl import DGLGraph, batch, unbatch
import dgl.function as DGLF
from .line_profiler_integration import profile
import os
import numpy as np

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 10

PAPER = os.getenv('PAPER', False)

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Note that during graph decoding they don't predict stereochemistry-related
# characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
# the 2-D graph first, then enumerate all possible 3-D forms and find the
# one with highest score.
def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))

def mol2dgl(cand_batch, mol_tree_batch):
    cand_graphs = []
    tree_mess_source_edges = [] # map these edges from trees to...
    tree_mess_target_edges = [] # these edges on candidate graphs
    tree_mess_target_nodes = []
    n_nodes = 0

    for mol, mol_tree, ctr_node_id in cand_batch:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        atom_x = []
        bond_x = []
        ctr_node = mol_tree.nodes_dict[ctr_node_id]
        ctr_bid = ctr_node['idx']
        g = DGLGraph()

        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            atom_x.append(atom_features(atom))
        g.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)

            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)

            x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            # Tree node ID in the batch
            x_bid = mol_tree.nodes_dict[x_nid - 1]['idx'] if x_nid > 0 else -1
            y_bid = mol_tree.nodes_dict[y_nid - 1]['idx'] if y_nid > 0 else -1
            if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                if mol_tree_batch.has_edge_between(x_bid, y_bid):
                    tree_mess_target_edges.append(
                            (begin_idx + n_nodes, end_idx + n_nodes))
                    tree_mess_source_edges.append((x_bid, y_bid))
                    tree_mess_target_nodes.append(end_idx + n_nodes)
                if mol_tree_batch.has_edge_between(y_bid, x_bid):
                    tree_mess_target_edges.append(
                            (end_idx + n_nodes, begin_idx + n_nodes))
                    tree_mess_source_edges.append((y_bid, x_bid))
                    tree_mess_target_nodes.append(begin_idx + n_nodes)
        g.add_edges(bond_src, bond_dst)

        n_nodes += n_atoms

        atom_x = cuda(torch.stack(atom_x, 0))
        bond_x = cuda(torch.stack(bond_x, 0))
        g.ndata['x'] = atom_x
        if n_bonds > 0:
            g.edata.update({
                'x': bond_x,
                'src_x': atom_x.new(n_bonds * 2, ATOM_FDIM).zero_(),
            })
        cand_graphs.append(g)

    return cand_graphs, tree_mess_source_edges, tree_mess_target_edges, \
           tree_mess_target_nodes


mpn_loopy_bp_msg = DGLF.copy_src(src='msg', out='msg')
mpn_loopy_bp_reduce = DGLF.sum(msg='msg', out='accum_msg')


class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, node):
        msg_input = node.data['msg_input']
        msg_delta = self.W_h(node.data['accum_msg'] + node.data['alpha'])
        msg = torch.relu(msg_input + msg_delta)
        return {'msg': msg}


if PAPER:
    mpn_gather_msg = [
        DGLF.copy_edge(edge='msg', out='msg'),
        DGLF.copy_edge(edge='alpha', out='alpha')
    ]
else:
    mpn_gather_msg = DGLF.copy_edge(edge='msg', out='msg')


if PAPER:
    mpn_gather_reduce = [
        DGLF.sum(msg='msg', out='m'),
        DGLF.sum(msg='alpha', out='accum_alpha'),
    ]
else:
    mpn_gather_reduce = DGLF.sum(msg='msg', out='m')


class GatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, node):
        if PAPER:
            #m = node['m']
            m = node.data['m'] + node.data['accum_alpha']
        else:
            m = node.data['m'] + node.data['alpha']
        return {
            'h': torch.relu(self.W_o(torch.cat([node.data['x'], m], 1))),
        }


class DGLJTMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        nn.Module.__init__(self)

        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)

        self.loopy_bp_updater = LoopyBPUpdate(hidden_size)
        self.gather_updater = GatherUpdate(hidden_size)
        self.hidden_size = hidden_size

        self.n_samples_total = 0
        self.n_nodes_total = 0
        self.n_edges_total = 0
        self.n_passes = 0

    def forward(self, cand_batch, mol_tree_batch):
        cand_graphs, tree_mess_src_edges, tree_mess_tgt_edges, tree_mess_tgt_nodes = \
                mol2dgl(cand_batch, mol_tree_batch)

        n_samples = len(cand_graphs)

        cand_graphs = batch(cand_graphs)
        cand_line_graph = cand_graphs.line_graph(backtracking=False, shared=True)

        n_nodes = cand_graphs.number_of_nodes()
        n_edges = cand_graphs.number_of_edges()

        cand_graphs = self.run(
                cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
                tree_mess_tgt_nodes, mol_tree_batch)

        # TODO: replace with unbatch or readout
        #cand_graphs = unbatch(cand_graphs)
        #g_repr = torch.stack([g.get_n_repr()['h'].mean(0) for g in cand_graphs], 0)
        n_repr = cand_graphs.pop_n_repr('h').split(cand_graphs.batch_num_nodes)
        g_repr = torch.stack([n_repr[i].mean(0) for i in range(n_samples)], 0)

        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1

        return g_repr

    def run(self, cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
            tree_mess_tgt_nodes, mol_tree_batch):
        n_nodes = cand_graphs.number_of_nodes()

        cand_graphs.update_edges(
            edge_func=lambda edges: {'src_x': edges.src['x']},
        )

        bond_features = cand_line_graph.ndata['x']
        source_features = cand_line_graph.ndata['src_x']
        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        cand_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': torch.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        zero_node_state = bond_features.new(n_nodes, self.hidden_size).zero_()
        cand_graphs.ndata.update({
            'm': zero_node_state.clone(),
            'h': zero_node_state.clone(),
        })

        if PAPER:
            cand_graphs.edata['alpha'] = \
                    cuda(torch.zeros(cand_graphs.number_of_edges(), self.hidden_size))

            src_u, src_v = zip(*tree_mess_src_edges)
            tgt_u, tgt_v = zip(*tree_mess_tgt_edges)
            alpha = mol_tree_batch.edges[src_u, src_v].data['m']
            cand_graphs.edges[tgt_u, tgt_v].data['alpha'] = alpha
        else:
            src_u, src_v = zip(*tree_mess_src_edges)
            alpha = mol_tree_batch.edges[src_u, src_v].data['m']
            node_idx = (torch.LongTensor(tree_mess_tgt_nodes)
                        .to(device=zero_node_state.device)[:, None]
                        .expand_as(alpha))
            node_alpha = zero_node_state.clone().scatter_add(0, node_idx, alpha)
            cand_graphs.ndata['alpha'] = node_alpha
            cand_graphs.update_edges(
                edge_func=lambda edges: {'alpha': edges.src['alpha']},
            )

        for i in range(self.depth - 1):
            cand_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.loopy_bp_updater,
            )

        cand_graphs.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        return cand_graphs
