import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .chemutils import get_mol
from networkx import Graph, DiGraph, line_graph, convert_node_labels_to_integers
from dgl import DGLGraph, batch, unbatch
import dgl.function as DGLF
from functools import partial
from .line_profiler_integration import profile

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return cuda(torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return cuda(torch.Tensor(fbond + fstereo))

@profile
def mol2graph(mol_batch):
    padding = cuda(torch.zeros(ATOM_FDIM + BOND_FDIM))
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        #mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append( atom_features(atom) )
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds) 
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)
        
        scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b

    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2

    return fatoms, fbonds, cuda(agraph), cuda(bgraph), scope

@profile
def mol2dgl(smiles_batch):
    n_nodes = 0
    graph_list = []
    for smiles in smiles_batch:
        atom_feature_list = []
        bond_feature_list = []
        bond_source_feature_list = []
        graph = DGLGraph()
        mol = get_mol(smiles)
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx())
            atom_feature_list.append(atom_features(atom))
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            features = bond_features(bond)
            graph.add_edge(begin_idx, end_idx)
            bond_feature_list.append(features)
            # set up the reverse direction
            graph.add_edge(end_idx, begin_idx)
            bond_feature_list.append(features)

        atom_x = torch.stack(atom_feature_list)
        graph.set_n_repr({'x': atom_x})
        if len(bond_feature_list) > 0:
            bond_x = torch.stack(bond_feature_list)
            graph.set_e_repr({
                'x': bond_x,
                'src_x': atom_x.new(len(bond_feature_list), ATOM_FDIM).zero_()
            })
        graph_list.append(graph)

    return graph_list


class MPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    @profile
    def forward(self, mol_graph):
        fatoms,fbonds,agraph,bgraph,scope = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = nn.ReLU()(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

# TODO: use SPMV
#def mpn_loopy_bp_msg(src, edge):
#    return src['msg']
mpn_loopy_bp_msg = DGLF.copy_src(src='msg', out='msg')


#def mpn_loopy_bp_reduce(node, msgs):
#    return {'accum_msg': torch.sum(msgs, 1)}
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


# TODO: can we use SPMV?
#def mpn_gather_msg(src, edge):
#    return edge['msg']
mpn_gather_msg = DGLF.copy_edge(edge='msg', out='msg')


#def mpn_gather_reduce(node, msgs):
#    return {'m': torch.sum(msgs, 1)}
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

    @profile
    def forward(self, mol_graph_list):
        n_samples = len(mol_graph_list)

        mol_graph = batch(mol_graph_list)
        mol_line_graph = line_graph(mol_graph, no_backtracking=True)

        n_nodes = len(mol_graph.nodes)
        n_edges = len(mol_graph.edges)

        mol_graph = self.run(mol_graph, mol_line_graph)
        mol_graph_list = unbatch(mol_graph)
        g_repr = torch.stack([g.get_n_repr()['h'].mean(0) for g in mol_graph_list], 0)

        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1

        return g_repr

    @profile
    def run(self, mol_graph, mol_line_graph):
        n_nodes = len(mol_graph.nodes)

        mol_graph.update_edge(
            #*zip(*mol_graph.edge_list),
            edge_func=lambda src, dst, edge: {'src_x': src['x']},
            batchable=True,
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
                True
            )

        mol_graph.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
            True
        )

        return mol_graph
