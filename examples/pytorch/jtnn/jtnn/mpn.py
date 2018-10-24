import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .chemutils import get_mol
from networkx import Graph, DiGraph, convert_node_labels_to_integers
from dgl import DGLGraph, batch, unbatch
import dgl.function as DGLF
from functools import partial
from .line_profiler_integration import profile
import numpy as np

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return (torch.Tensor(fbond + fstereo))

def mol2dgl(smiles_batch):
    n_nodes = 0
    graph_list = []
    for smiles in smiles_batch:
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        atom_x = []
        bond_x = []
        graph = DGLGraph()
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            atom_x.append(atom_features(atom))
        graph.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            features = bond_features(bond)
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            # set up the reverse direction
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)
        graph.add_edges(bond_src, bond_dst)

        atom_x = cuda(torch.stack(atom_x, 0))
        bond_x = cuda(torch.stack(bond_x, 0))
        graph.set_n_repr({'x': atom_x})
        if n_bonds > 0:
            graph.set_e_repr({
                'x': bond_x,
                'src_x': atom_x.new(n_bonds * 2, ATOM_FDIM).zero_()
            })
        graph_list.append(graph)

    return graph_list


mpn_loopy_bp_msg = DGLF.copy_src(src='msg', out='msg')
mpn_loopy_bp_reduce = DGLF.sum(msg='msg', out='accum_msg')


class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, node):
        msg_input = node['msg_input']
        msg_delta = self.W_h(node['accum_msg'])
        msg = F.relu(msg_input + msg_delta)
        return {'msg': msg}


mpn_gather_msg = DGLF.copy_edge(edge='msg', out='msg')
mpn_gather_reduce = DGLF.sum(msg='msg', out='m')


class GatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, node):
        m = node['m']
        return {
            'h': F.relu(self.W_o(torch.cat([node['x'], m], 1))),
        }


class DGLMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(DGLMPN, self).__init__()

        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)

        self.loopy_bp_updater = LoopyBPUpdate(hidden_size)
        self.gather_updater = GatherUpdate(hidden_size)
        self.hidden_size = hidden_size

        self.n_samples_total = 0
        self.n_nodes_total = 0
        self.n_edges_total = 0
        self.n_passes = 0

    def forward(self, mol_graph_list):
        n_samples = len(mol_graph_list)

        mol_graph = batch(mol_graph_list)
        mol_line_graph = mol_graph.line_graph(backtracking=False, shared=True)

        n_nodes = mol_graph.number_of_nodes()
        n_edges = mol_graph.number_of_edges()

        mol_graph = self.run(mol_graph, mol_line_graph)
        mol_graph_list = unbatch(mol_graph)
        g_repr = torch.stack([g.get_n_repr()['h'].mean(0) for g in mol_graph_list], 0)

        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1

        return g_repr

    def run(self, mol_graph, mol_line_graph):
        n_nodes = mol_graph.number_of_nodes()

        mol_graph.update_edge(
            edge_func=lambda src, dst, edge: {'src_x': src['x']},
        )

        bond_features = mol_line_graph.get_n_repr()['x']
        source_features = mol_line_graph.get_n_repr()['src_x']

        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        mol_line_graph.set_n_repr({
            'msg_input': msg_input,
            'msg': F.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        mol_graph.set_n_repr({
            'm': bond_features.new(n_nodes, self.hidden_size).zero_(),
            'h': bond_features.new(n_nodes, self.hidden_size).zero_(),
        })

        for i in range(self.depth - 1):
            mol_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.loopy_bp_updater,
            )

        mol_graph.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        return mol_graph
