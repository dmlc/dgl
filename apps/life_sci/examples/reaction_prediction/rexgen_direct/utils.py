import dgl
import errno
import numpy as np
import os
import random
import torch

from collections import defaultdict
from rdkit import Chem

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def setup(args, seed=0):
    """Setup for the experiment:

    1. Decide whether to use CPU or GPU for training
    2. Fix random seed for python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed to use.

    Returns
    -------
    args
        Updated configuration
    """
    assert args['max_k'] >= max(args['top_ks']), \
        'Expect max_k to be no smaller than the possible options ' \
        'of top_ks, got {:d} and {:d}'.format(args['max_k'], max(args['top_ks']))

    if torch.cuda.is_available():
        args['device'] = 'cuda:0'
    else:
        args['device'] = 'cpu'

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    mkdir_p(args['result_path'])

    return args

def collate(data):
    """Collate multiple datapoints

    Parameters
    ----------
    data : list of 7-tuples
        Each tuple is for a single datapoint, consisting of
        a reaction, graph edits in the reaction, an RDKit molecule instance for all reactants,
        a DGLGraph for all reactants, a complete graph for all reactants, the features for each
        pair of atoms and the labels for each pair of atoms.

    Returns
    -------
    reactions : list of str
        List of reactions.
    graph_edits : list of str
        List of graph edits in the reactions.
    mols : list of rdkit.Chem.rdchem.Mol
        List of RDKit molecule instances for the reactants.
    batch_mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs.
    batch_complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs.
    batch_atom_pair_labels : float32 tensor of shape (V, 10)
        Labels of atom pairs in the batch of graphs.
    """
    reactions, graph_edits, mols, mol_graphs, complete_graphs, \
    atom_pair_feats, atom_pair_labels = map(list, zip(*data))

    batch_mol_graphs = dgl.batch(mol_graphs)
    batch_mol_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_mol_graphs.set_e_initializer(dgl.init.zero_initializer)

    batch_complete_graphs = dgl.batch(complete_graphs)
    batch_complete_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.set_e_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.edata['feats'] = torch.cat(atom_pair_feats, dim=0)

    batch_atom_pair_labels = torch.cat(atom_pair_labels, dim=0)

    return reactions, graph_edits, mols, batch_mol_graphs, \
           batch_complete_graphs, batch_atom_pair_labels

def reaction_center_prediction(device, model, mol_graphs, complete_graphs):
    """Perform a soft prediction on reaction center.

    Parameters
    ----------
    device : str
        Device to use for computation, e.g. 'cpu', 'cuda:0'
    model : nn.Module
        Model for prediction.
    mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs
    complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs

    Returns
    -------
    scores : float32 tensor of shape (E_full, 5)
        Predicted scores for each pair of atoms to perform one of the following
        5 actions in reaction:

        * The bond between them gets broken
        * Forming a single bond
        * Forming a double bond
        * Forming a triple bond
        * Forming an aromatic bond
    biased_scores : float32 tensor of shape (E_full, 5)
        Comparing to scores, a bias is added if the pair is for a same atom.
    """
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    edge_feats = mol_graphs.edata.pop('he').to(device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)

    return model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)

def rough_eval(complete_graphs, preds, labels, num_correct):
    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        end = start + complete_graphs.batch_num_edges[i]
        preds_i = preds[start:end, :].flatten()
        labels_i = labels[start:end, :].flatten()
        for k in num_correct.keys():
            topk_values, topk_indices = torch.topk(preds_i, k)
            is_correct = labels_i[topk_indices].sum() == labels_i.sum().float().cpu().data.item()
            num_correct[k].append(is_correct)
        start = end

def rough_eval_on_a_loader(args, model, data_loader):
    """A rough evaluation of model performance in the middle of training.

    For final evaluation, we will eliminate some possibilities based on prior knowledge.

    Parameters
    ----------
    args : dict
        Configurations fot the experiment.
    model : nn.Module
        Model for reaction center prediction.
    data_loader : torch.utils.data.DataLoader
        Loader for fetching and batching data.

    Returns
    -------
    str
        Message for evluation result.
    """
    model.eval()
    num_correct = {k: [] for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        rough_eval(batch_complete_graphs, biased_pred, batch_atom_pair_labels, num_correct)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, np.mean(correct_count))

    return msg

def eval(complete_graphs, preds, reactions, graph_edits, num_correct, max_k, easy):
    """Evaluate top-k accuracies for reaction center prediction.

    Parameters
    ----------
    complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs
    preds : float32 tensor of shape (E_full, 5)
        Soft predictions for reaction center, E_full being the number of possible
        atom-pairs and 5 being the number of possible bond changes
    reactions : list of str
        List of reactions.
    graph_edits : list of str
        List of graph edits in the reactions.
    num_correct : dict
        Counting the number of datapoints for meeting top-k accuracies.
    max_k : int
        Maximum number of atom pairs to be selected. This is intended to be larger
        than max(num_correct.keys()) as we will filter out many atom pairs due to
        considerations such as avoiding duplicates.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.
    """
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
    id_to_bond_change = {v: k for k, v in bond_change_to_id.items()}
    num_change_types = len(bond_change_to_id)

    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        # Decide which atom-pairs will be considered.
        reaction_i = reactions[i]
        reaction_atoms_i = []
        reaction_bonds_i = defaultdict(bool)
        reactants_i, _, product_i = reaction_i.split('>')
        product_mol_i = Chem.MolFromSmiles(product_i)
        product_atoms_i = set([atom.GetAtomMapNum() for atom in product_mol_i.GetAtoms()])

        for reactant in reactants_i.split('.'):
            reactant_mol = Chem.MolFromSmiles(reactant)
            reactant_atoms = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
            if (len(set(reactant_atoms) & product_atoms_i) > 0) or (not easy):
                reaction_atoms_i.extend(reactant_atoms)
                for bond in reactant_mol.GetBonds():
                    end_atoms = sorted([bond.GetBeginAtom().GetAtomMapNum(),
                                        bond.GetEndAtom().GetAtomMapNum()])
                    bond = tuple(end_atoms + [bond.GetBondTypeAsDouble()])
                    reaction_bonds_i[bond] = True

        num_nodes = complete_graphs.batch_num_nodes[i]
        end = start + complete_graphs.batch_num_edges[i]
        preds_i = preds[start:end, :].flatten()
        candidate_bonds = []
        topk_values, topk_indices = torch.topk(preds_i, max_k)
        for j in range(max_k):
            preds_i_j = topk_indices[j].cpu().item()
            # A bond change can be either losing the bond or forming a
            # single, double, triple or aromatic bond
            change_id = preds_i_j % num_change_types
            change_type = id_to_bond_change[change_id]
            pair_id = preds_i_j // num_change_types
            atom1 = pair_id // num_nodes + 1
            atom2 = pair_id % num_nodes + 1
            # Avoid duplicates and an atom cannot form a bond with itself
            if atom1 >= atom2:
                continue
            if atom1 not in reaction_atoms_i:
                continue
            if atom2 not in reaction_atoms_i:
                continue
            candidate = (int(atom1), int(atom2), float(change_type))
            if reaction_bonds_i[candidate]:
                continue
            candidate_bonds.append(candidate)

        gold_bonds = []
        gold_edits = graph_edits[i]
        for edit in gold_edits.split(';'):
            atom1, atom2, change_type = edit.split('-')
            atom1, atom2 = int(atom1), int(atom2)
            gold_bonds.append((min(atom1, atom2), max(atom1, atom2), float(change_type)))

        for k in num_correct.keys():
            if set(gold_bonds) <= set(candidate_bonds[:k]):
                num_correct[k] += 1
        start = end

def reaction_center_final_eval(args, model, data_loader, easy):
    """Final evaluation of model performance.

    args : dict
        Configurations fot the experiment.
    model : nn.Module
        Model for reaction center prediction.
    data_loader : torch.utils.data.DataLoader
        Loader for fetching and batching data.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.

    Returns
    -------
    msg : str
        Summary of the top-k evaluation.
    """
    model.eval()
    num_correct = {k: 0 for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        eval(batch_complete_graphs, biased_pred, batch_reactions,
             batch_graph_edits, num_correct, args['max_k'], easy)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, correct_count / len(data_loader.dataset))

    return msg
