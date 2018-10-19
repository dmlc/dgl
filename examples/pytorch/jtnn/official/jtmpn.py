import torch
import torch.nn as nn
from .nnutils import create_var, index_select_ND, cuda
from .chemutils import get_mol
#from mpn import atom_features, bond_features, ATOM_FDIM, BOND_FDIM
import rdkit.Chem as Chem
from dgl import DGLGraph, batch, unbatch
import dgl.function as DGLF
from .line_profiler_integration import profile
import os

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
    return cuda(torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    return cuda(torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))


class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    @profile
    def forward(self, cand_batch, tree_mess):
        fatoms,fbonds = [],[] 
        in_bonds,all_bonds = [],[] 
        mess_dict,all_mess = {},[create_var(cuda(torch.zeros(self.hidden_size)))] #Ensure index 0 is vec(0)
        total_atoms = 0
        scope = []

        for e,vec in tree_mess.items():
            mess_dict[e] = len(all_mess)
            all_mess.append(vec)

        for mol,all_nodes,ctr_node in cand_batch:
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([]) 
        
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = len(all_mess) + len(all_bonds)  #bond idx offseted by len(all_mess)
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                in_bonds[x].append(b)

                # FIXME: https://github.com/wengong-jin/icml18-jtnn/issues/19
                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid,y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid,y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid,x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid,x_bid)]
                        in_bonds[x].append(mess_idx)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms
        
        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by len(all_mess)
                if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
                    bgraph[b1,i] = b2

        atom_hiddens = self.run(fatoms, fbonds, agraph, bgraph, tree_message)
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    @profile
    def run(self, fatoms, fbonds, agraph, bgraph, tree_message):
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(cuda(agraph))
        bgraph = create_var(cuda(bgraph))

        binput = self.W_i(fbonds)
        graph_message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message,graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = nn.ReLU()(binput + nei_message)

        message = torch.cat([tree_message,graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))

        return atom_hiddens


@profile
def mol2dgl(cand_batch, mol_tree_batch):
    cand_graphs = []
    tree_mess_source_edges = [] # map these edges from trees to...
    tree_mess_target_edges = [] # these edges on candidate graphs
    tree_mess_target_nodes = []
    n_nodes = 0

    for mol, mol_tree, ctr_node_id in cand_batch:
        atom_feature_list = []
        bond_feature_list = []
        ctr_node = mol_tree.nodes[ctr_node_id]
        ctr_bid = ctr_node['idx']
        g = DGLGraph()

        for atom in mol.GetAtoms():
            atom_feature_list.append(atom_features(atom))
            g.add_node(atom.GetIdx())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)

            g.add_edge(begin_idx, end_idx)
            bond_feature_list.append(features)
            g.add_edge(end_idx, begin_idx)
            bond_feature_list.append(features)

            x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            # Tree node ID in the batch
            x_bid = mol_tree.nodes[x_nid - 1]['idx'] if x_nid > 0 else -1
            y_bid = mol_tree.nodes[y_nid - 1]['idx'] if y_nid > 0 else -1
            if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                if (x_bid, y_bid) in mol_tree_batch.edge_list:
                    tree_mess_target_edges.append(
                            (begin_idx + n_nodes, end_idx + n_nodes))
                    tree_mess_source_edges.append((x_bid, y_bid))
                    tree_mess_target_nodes.append(end_idx + n_nodes)
                if (y_bid, x_bid) in mol_tree_batch.edge_list:
                    tree_mess_target_edges.append(
                            (end_idx + n_nodes, begin_idx + n_nodes))
                    tree_mess_source_edges.append((y_bid, x_bid))
                    tree_mess_target_nodes.append(begin_idx + n_nodes)

        n_nodes += len(g.nodes)

        atom_x = torch.stack(atom_feature_list)
        g.set_n_repr({'x': atom_x})
        if len(bond_feature_list) > 0:
            bond_x = torch.stack(bond_feature_list)
            g.set_e_repr({
                'x': bond_x,
                'src_x': atom_x.new(len(bond_feature_list), ATOM_FDIM).zero_()
            })
        cand_graphs.append(g)

    return cand_graphs, tree_mess_source_edges, tree_mess_target_edges, \
           tree_mess_target_nodes


# TODO: use SPMV
mpn_loopy_bp_msg = DGLF.copy_src(src='msg', out='msg')
#def mpn_loopy_bp_msg(src, edge):
#    return src['msg']


mpn_loopy_bp_reduce = DGLF.sum(msg='msg', out='accum_msg')
#def mpn_loopy_bp_reduce(node, msgs):
#    return {'accum_msg': torch.sum(msgs, 1)}


class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, node):
        msg_input = node['msg_input']
        msg_delta = self.W_h(node['accum_msg'] + node['alpha'])
        msg = torch.relu(msg_input + msg_delta)
        return {'msg': msg}


# TODO: can we use SPMV?
#def mpn_gather_msg(src, edge):
#    if PAPER:
#        return {'msg': edge['msg'], 'alpha': edge['alpha']}
#    else:
#        return {'msg': edge['msg']}
if PAPER:
    mpn_gather_msg = [
        DGLF.copy_edge(edge='msg', out='msg'),
        DGLF.copy_edge(edge='alpha', out='alpha')
    ]
else:
    mpn_gather_msg = DGLF.copy_edge(edge='msg', out='msg')


#def mpn_gather_reduce(node, msgs):
#    if PAPER:
#        return {'m': torch.sum(msgs['msg'], 1) + torch.sum(msgs['alpha'], 1)}
#    else:
#        return {'m': torch.sum(msgs['msg'], 1)}
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
            m = node['m'] + node['accum_alpha']
        else:
            m = node['m'] + node['alpha']
        return {
            'h': torch.relu(self.W_o(torch.cat([node['x'], m], 1))),
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

    @profile
    def forward(self, cand_batch, mol_tree_batch):
        cand_graphs, tree_mess_src_edges, tree_mess_tgt_edges, tree_mess_tgt_nodes = \
                mol2dgl(cand_batch, mol_tree_batch)

        n_samples = len(cand_graphs)

        cand_graphs = batch(cand_graphs)
        cand_line_graph = line_graph(cand_graphs, no_backtracking=True)

        n_nodes = len(cand_graphs.nodes)
        n_edges = len(cand_graphs.edges)

        cand_graphs = self.run(
                cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
                tree_mess_tgt_nodes, mol_tree_batch)

        cand_graphs = unbatch(cand_graphs)
        g_repr = torch.stack([g.get_n_repr()['h'].mean(0) for g in cand_graphs], 0)

        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1

        return g_repr

    @profile
    def run(self, cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
            tree_mess_tgt_nodes, mol_tree_batch):
        n_nodes = len(cand_graphs.nodes)

        cand_graphs.update_edge(
            #*zip(*cand_graphs.edge_list),
            edge_func=lambda src, dst, edge: {'src_x': src['x']},
            batchable=True,
        )

        bond_features = cand_line_graph.get_n_repr()['x']
        source_features = cand_line_graph.get_n_repr()['src_x']
        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        cand_line_graph.set_n_repr({
            'msg_input': msg_input,
            'msg': torch.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        zero_node_state = bond_features.new(n_nodes, self.hidden_size).zero_()
        cand_graphs.set_n_repr({
            'm': zero_node_state.clone(),
            'h': zero_node_state.clone(),
        })

        # TODO: context
        if PAPER:
            cand_graphs.set_e_repr({
                'alpha': cuda(torch.zeros(len(cand_graphs.edge_list), self.hidden_size))
            })

            alpha = mol_tree_batch.get_e_repr(*zip(*tree_mess_src_edges))['m']
            cand_graphs.set_e_repr({'alpha': alpha}, *zip(*tree_mess_tgt_edges))
        else:
            alpha = mol_tree_batch.get_e_repr(*zip(*tree_mess_src_edges))['m']
            node_idx = (torch.LongTensor(tree_mess_tgt_nodes)
                        .to(device=zero_node_state.device)[:, None]
                        .expand_as(alpha))
            node_alpha = zero_node_state.clone().scatter_add(0, node_idx, alpha)
            cand_graphs.set_n_repr({'alpha': node_alpha})
            cand_graphs.update_edge(
                #*zip(*cand_graphs.edge_list),
                edge_func=lambda src, dst, edge: {'alpha': src['alpha']},
                batchable=True,
            )

        for i in range(self.depth - 1):
            cand_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.loopy_bp_updater,
                True
            )

        cand_graphs.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
            True
        )

        return cand_graphs
