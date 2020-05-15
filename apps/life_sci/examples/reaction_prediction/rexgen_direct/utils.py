import dgl
import errno
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn

from collections import defaultdict
from copy import deepcopy
from dgllife.data import USPTOCenter, WLNCenterDataset
from dgllife.model import load_pretrained, WLNReactionCenter
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from molvs import Standardizer
except ImportError as e:
    print('MolVS is not installed, which is required for candidate ranking')

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

def set_seed(seed=0):
    """Fix random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def count_parameters(model):
    """Get the number of trainable parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The model

    Returns
    -------
    int
        Number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_center_subset(dataset, subset_id, num_subsets):
    """Get subset for reaction center identification.

    Parameters
    ----------
    dataset : WLNCenterDataset
        Dataset for reaction center prediction with WLN
    subset_id : int
        Index for the subset
    num_subsets : int
        Number of total subsets
    """
    if num_subsets == 1:
        return

    total_size = len(dataset)
    subset_size = total_size // num_subsets
    start = subset_id * subset_size
    end = (subset_id + 1) * subset_size

    dataset.mols = dataset.mols[start:end]
    dataset.reactions = dataset.reactions[start:end]
    dataset.graph_edits = dataset.graph_edits[start:end]
    dataset.reactant_mol_graphs = dataset.reactant_mol_graphs[start:end]
    dataset.atom_pair_features = [None for _ in range(subset_size)]
    dataset.atom_pair_labels = [None for _ in range(subset_size)]

class Optimizer(nn.Module):
    """Wrapper for optimization

    Parameters
    ----------
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    num_accum_times : int
        Number of times for accumulating gradients
    max_grad_norm : float or None
        If not None, gradient clipping will be performed
    """
    def __init__(self, model, lr, optimizer, num_accum_times=1, max_grad_norm=None):
        super(Optimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.step_count = 0
        self.num_accum_times = num_accum_times
        self.max_grad_norm = max_grad_norm
        self._reset()

    def _reset(self):
        self.optimizer.zero_grad()

    def _clip_grad_norm(self):
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(self.model.parameters(),
                                        self.max_grad_norm)
        return grad_norm

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only

        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        self.step_count += 1
        loss.backward()
        if self.step_count % self.num_accum_times == 0:
            grad_norm = self._clip_grad_norm()
            self.optimizer.step()
            self._reset()

            return grad_norm
        else:
            return 0

    def decay_lr(self, decay_rate):
        """Decay learning rate.

        Parameters
        ----------
        decay_rate : float
            Multiply the current learning rate by the decay_rate
        """
        self.lr *= decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

class MultiProcessOptimizer(Optimizer):
    """Wrapper for optimization with multiprocess

    Parameters
    ----------
    n_processes : int
        Number of processes used
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    max_grad_norm : float or None
        If not None, gradient clipping will be performed.
    """
    def __init__(self, n_processes, model, lr, optimizer, max_grad_norm=None):
        super(MultiProcessOptimizer, self).__init__(lr=lr, model=model, optimizer=optimizer,
                                                    max_grad_norm=max_grad_norm)
        self.n_processes = n_processes

    def _sync_gradient(self):
        """Average gradients across all subprocesses."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.n_processes

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only

        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        loss.backward()
        self._sync_gradient()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self._reset()

        return grad_norm

def synchronize(num_gpus):
    """Synchronize all processes for multi-gpu training.

    Parameters
    ----------
    num_gpus : int
        Number of gpus used
    """
    if num_gpus > 1:
        dist.barrier()

def collate_center(data):
    """Collate multiple datapoints for reaction center prediction

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
    batch_mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs.
    batch_complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs.
    batch_atom_pair_labels : float32 tensor of shape (V, 10)
        Labels of atom pairs in the batch of graphs.
    """
    reactions, graph_edits, mol_graphs, complete_graphs, \
    atom_pair_feats, atom_pair_labels = map(list, zip(*data))

    batch_mol_graphs = dgl.batch(mol_graphs)
    batch_mol_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_mol_graphs.set_e_initializer(dgl.init.zero_initializer)

    batch_complete_graphs = dgl.batch(complete_graphs)
    batch_complete_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.set_e_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.edata['feats'] = torch.cat(atom_pair_feats, dim=0)

    batch_atom_pair_labels = torch.cat(atom_pair_labels, dim=0)

    return reactions, graph_edits, batch_mol_graphs, \
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

bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
id_to_bond_change = {v: k for k, v in bond_change_to_id.items()}
num_change_types = len(bond_change_to_id)

def get_candidate_bonds(reaction, preds, num_nodes, max_k, easy, include_scores=False):
    """Get candidate bonds for a reaction.

    Parameters
    ----------
    reaction : str
        Reaction
    preds : float32 tensor of shape (E * 5)
        E for the number of edges in a complete graph and 5 for the number of possible
        bond changes.
    num_nodes : int
        Number of nodes in the graph.
    max_k : int
        Maximum number of atom pairs to be selected.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.
    include_scores : bool
        Whether to include the scores for the atom pairs selected. Default to False.

    Returns
    -------
    list of 3-tuples or 4-tuples
        The first three elements in a tuple separately specify the first atom,
        the second atom and the type for bond change. If include_scores is True,
        the score for the prediction will be included as a fourth element.
    """
    # Decide which atom-pairs will be considered.
    reaction_atoms = []
    reaction_bonds = defaultdict(bool)
    reactants, _, product = reaction.split('>')
    product_mol = Chem.MolFromSmiles(product)
    product_atoms = set([atom.GetAtomMapNum() for atom in product_mol.GetAtoms()])

    for reactant in reactants.split('.'):
        reactant_mol = Chem.MolFromSmiles(reactant)
        reactant_atoms = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
        # In the hard mode, all reactant atoms will be included.
        # In the easy mode, only reactants contributing atoms to the product will be included.
        if (len(set(reactant_atoms) & product_atoms) > 0) or (not easy):
            reaction_atoms.extend(reactant_atoms)
            for bond in reactant_mol.GetBonds():
                end_atoms = sorted([bond.GetBeginAtom().GetAtomMapNum(),
                                    bond.GetEndAtom().GetAtomMapNum()])
                bond = tuple(end_atoms + [bond.GetBondTypeAsDouble()])
                # Bookkeep bonds already in reactants
                reaction_bonds[bond] = True

    candidate_bonds = []
    topk_values, topk_indices = torch.topk(preds, max_k)
    for j in range(max_k):
        preds_j = topk_indices[j].cpu().item()
        # A bond change can be either losing the bond or forming a
        # single, double, triple or aromatic bond
        change_id = preds_j % num_change_types
        change_type = id_to_bond_change[change_id]
        pair_id = preds_j // num_change_types
        # Atom map numbers
        atom1 = pair_id // num_nodes + 1
        atom2 = pair_id % num_nodes + 1
        # Avoid duplicates and an atom cannot form a bond with itself
        if atom1 >= atom2:
            continue
        if atom1 not in reaction_atoms:
            continue
        if atom2 not in reaction_atoms:
            continue
        candidate = (int(atom1), int(atom2), float(change_type))
        if reaction_bonds[candidate]:
            continue
        if include_scores:
            candidate += (float(topk_values[j].cpu().item()),)
        candidate_bonds.append(candidate)

    return candidate_bonds

def reaction_center_eval(complete_graphs, preds, reactions,
                         graph_edits, num_correct, max_k, easy):
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
    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        end = start + complete_graphs.batch_num_edges[i]
        candidate_bonds = get_candidate_bonds(
            reactions[i], preds[start:end, :].flatten(),
            complete_graphs.batch_num_nodes[i], max_k, easy)

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

def reaction_center_final_eval(args, top_ks, model, data_loader, easy):
    """Final evaluation of model performance.

    args : dict
        Configurations fot the experiment.
    top_ks : list of int
        Options for top-k evaluation
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
    num_correct = {k: 0 for k in top_ks}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        reaction_center_eval(batch_complete_graphs, biased_pred, batch_reactions,
             batch_graph_edits, num_correct, args['max_k'], easy)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, correct_count / len(data_loader.dataset))

    return msg + '\n'

def output_candidate_bonds_for_a_reaction(info, max_k):
    """Prepare top-k atom pairs for each reaction as candidate bonds

    Parameters
    ----------
    info : 3-tuple for a reaction
        Consists of the reaction, the scores for atom-pairs in reactants
        and the number of nodes in reactants.
    max_k : int
        Maximum number of atom pairs to be selected.

    Returns
    -------
    candidate_string : str
        String representing candidate bonds for a reaction. Each candidate
        bond is of format 'atom1 atom2 change_type score'.
    """
    reaction, preds, num_nodes = info
    # Note that we use the easy mode by default, which is also the
    # setting in the paper.
    candidate_bonds = get_candidate_bonds(reaction, preds, num_nodes, max_k,
                                          easy=True, include_scores=True)
    candidate_string = ''
    for candidate in candidate_bonds:
        # A 4-tuple consisting of the atom mapping number of atom 1,
        # atom 2, the bond change type and the score
        candidate_string += '{} {} {:.1f} {:.3f};'.format(
            candidate[0], candidate[1], candidate[2], candidate[3])
    candidate_string += '\n'

    return candidate_string

def prepare_reaction_center(args, reaction_center_config):
    """Use a trained model for reaction center prediction to prepare candidate bonds.

    Parameters
    ----------
    args : dict
        Configuration for the experiment.
    reaction_center_config : dict
        Configuration for the experiment on reaction center prediction.

    Returns
    -------
    path_to_candidate_bonds : dict
        Mapping 'train', 'val', 'test' to the corresponding files for candidate bonds.
    """
    if args['center_model_path'] is None:
        reaction_center_model = load_pretrained('wln_center_uspto').to(args['device'])
    else:
        reaction_center_model = WLNReactionCenter(
            node_in_feats=reaction_center_config['node_in_feats'],
            edge_in_feats=reaction_center_config['edge_in_feats'],
            node_pair_in_feats=reaction_center_config['node_pair_in_feats'],
            node_out_feats=reaction_center_config['node_out_feats'],
            n_layers=reaction_center_config['n_layers'],
            n_tasks=reaction_center_config['n_tasks'])
        reaction_center_model.load_state_dict(
            torch.load(args['center_model_path'])['model_state_dict'])
        reaction_center_model = reaction_center_model.to(args['device'])
    reaction_center_model.eval()

    path_to_candidate_bonds = dict()
    for subset in ['train', 'val', 'test']:
        if '{}_path'.format(subset) not in args:
            continue

        path_to_candidate_bonds[subset] = args['result_path'] + \
                                          '/{}_candidate_bonds.txt'.format(subset)
        if os.path.isfile(path_to_candidate_bonds[subset]):
            continue

        print('Processing subset {}...'.format(subset))
        print('Stage 1/3: Loading dataset...')
        if args['{}_path'.format(subset)] is None:
            dataset = USPTOCenter(subset, num_processes=args['num_processes'])
        else:
            dataset = WLNCenterDataset(raw_file_path=args['{}_path'.format(subset)],
                                       mol_graph_path='{}.bin'.format(subset),
                                       num_processes=args['num_processes'])

        dataloader = DataLoader(dataset, batch_size=args['reaction_center_batch_size'],
                                collate_fn=collate_center, shuffle=False)

        print('Stage 2/3: Performing model prediction...')
        output_strings = []
        for batch_id, batch_data in enumerate(dataloader):
            print('Computing candidate bonds for batch {:d}/{:d}'.format(
                batch_id + 1, len(dataloader)))
            batch_reactions, batch_graph_edits, batch_mol_graphs, \
            batch_complete_graphs, batch_atom_pair_labels = batch_data
            with torch.no_grad():
                pred, biased_pred = reaction_center_prediction(
                    args['device'], reaction_center_model,
                    batch_mol_graphs, batch_complete_graphs)
            batch_size = len(batch_reactions)
            start = 0
            for i in range(batch_size):
                end = start + batch_complete_graphs.batch_num_edges[i]
                output_strings.append(output_candidate_bonds_for_a_reaction(
                    (batch_reactions[i], biased_pred[start:end, :].flatten(),
                    batch_complete_graphs.batch_num_nodes[i]), reaction_center_config['max_k']
                ))
                start = end

        print('Stage 3/3: Output candidate bonds...')
        with open(path_to_candidate_bonds[subset], 'w') as f:
            for candidate_string in output_strings:
                f.write(candidate_string)

        del dataset
        del dataloader

    del reaction_center_model

    return path_to_candidate_bonds

def collate_rank_train(data):
    """Collate multiple datapoints for candidate product ranking during training

    Parameters
    ----------
    data : list of 3-tuples
        Each tuple is for a single datapoint, consisting of DGLGraphs for reactants and candidate
        products, scores for candidate products by the model for reaction center prediction,
        and labels for candidate products.

    Returns
    -------
    batch_reactant_graphs : DGLGraph
        DGLGraph for a batch of batch_size reactants.
    product_graphs : DGLGraph
        DGLGraph for a batch of B candidate products
    combo_scores : float32 tensor of shape (B, 1)
        Scores for candidate products by the model for reaction center prediction.
    labels : int64 tensor of shape (N, 1)
        Indices for the true candidate product across reactions, which is always 0
        with pre-processing. N is for the number of reactions.
    batch_num_candidate_products : list of int
        Number of candidate products for the reactions in this batch.
    """
    batch_graphs, batch_combo_scores, batch_labels = map(list, zip(*data))
    batch_reactant_graphs = dgl.batch([g_list[0] for g_list in batch_graphs])

    batch_num_candidate_products = []
    batch_product_graphs = []
    for g_list in batch_graphs:
        batch_num_candidate_products.append(len(g_list) - 1)
        batch_product_graphs.extend(g_list[1:])
    batch_product_graphs = dgl.batch(batch_product_graphs)
    batch_combo_scores = torch.cat(batch_combo_scores, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)

    return batch_reactant_graphs, batch_product_graphs, batch_combo_scores, batch_labels, \
           batch_num_candidate_products

def collate_rank_eval(data):
    """Collate multiple datapoints for candidate product ranking during evaluation

    Parameters
    ----------
    data : list of 3-tuples
        Each tuple is for a single datapoint, consisting of DGLGraphs for reactants and candidate
        products, scores for candidate products by the model for reaction center prediction,
        and valid combos of candidate bond changes, one for each candidate product.

    Returns
    -------
    batch_reactant_graph : DGLGraph
        DGLGraph for a batch of batch_size reactants.
        None will be returned if no valid candidate products exist.
    batch_product_graphs : DGLGraph
        DGLGraph for a batch of B candidate products.
        None will be returned if no valid candidate products exist.
    batch_combo_scores : float32 tensor of shape (B, 1)
        Scores for candidate products by the model for reaction center prediction.
        None will be returned if no valid candidate products exist.
    valid_candidate_combos_list : list of list
        valid_candidate_combos_list[i] gives valid combos of candidate bond changes for the
        i-th reaction. valid_candidate_combos_list[i][j] gives a list of tuples, which is
        the j-th valid combo of candidate bond changes for the reaction. Each tuple is of form
        (atom1, atom2, change_type, score). atom1, atom2 are the atom mapping numbers - 1 of the
        two end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
        a single, double, triple, and aromatic bond. None will be returned if no valid candidate
        products exist.
    reactant_mols_list : list of rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants in the batch.
        None will be returned if no valid candidate products exist.
    real_bond_changes_list : list of list
        real_bond_changes_list[i] gives the ground truth bond changes in the i-th reaction,
        which is a list of tuples. Each tuple is of form (atom1, atom2, change_type). atom1,
        atom2 are the atom mapping numbers - 1 of the two end atoms. change_type can be
        0, 1, 2, 3, 1.5, separately for losing a bond, forming a single, double, triple, and
        aromatic bond. None will be returned if no valid candidate products exist.
    product_mols_list : list of rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the candidate products in each reaction.
        None will be returned if no valid candidate products exist.
    batch_num_candidate_products : list of int
        Number of candidate products for the reactions in this batch.
    """
    batch_graphs, batch_combo_scores, batch_valid_candidate_combos, \
    batch_reactant_mols, batch_real_bond_changes, batch_product_mols = map(list, zip(*data))

    batch_reactant_graphs = []
    batch_product_graphs = []
    combo_scores_list = []
    valid_candidate_combos_list = []
    reactant_mols_list = []
    real_bond_changes_list = []
    product_mols_list = []
    batch_num_candidate_products = []

    for i in range(len(batch_graphs)):
        g_list = batch_graphs[i]
        # No valid candidate products have been predicted
        if len(g_list) == 1:
            continue
        batch_reactant_graphs.append(g_list[0])
        batch_product_graphs.extend(g_list[1:])
        combo_scores_list.append(batch_combo_scores[i])
        valid_candidate_combos_list.append(batch_valid_candidate_combos[i])
        reactant_mols_list.append(batch_reactant_mols[i])
        real_bond_changes_list.append(batch_real_bond_changes[i])
        product_mols_list.append(batch_product_mols[i])
        batch_num_candidate_products.append(len(g_list) - 1)

    if len(batch_product_graphs) == 0:
        return None, None, None, None, None, None, None, None

    batch_reactant_graphs = dgl.batch(batch_reactant_graphs)
    batch_product_graphs = dgl.batch(batch_product_graphs)
    batch_combo_scores = torch.cat(combo_scores_list, dim=0)

    return batch_reactant_graphs, batch_product_graphs, batch_combo_scores, \
           valid_candidate_combos_list, reactant_mols_list, real_bond_changes_list, \
           product_mols_list, batch_num_candidate_products

def sanitize_smiles_molvs(smiles, largest_fragment=False):
    """Sanitize a SMILES with MolVS

    Parameters
    ----------
    smiles : str
        SMILES string for a molecule.
    largest_fragment : bool
        Whether to select only the largest covalent unit in a molecule with
        multiple fragments. Default to False.

    Returns
    -------
    str
        SMILES string for the sanitized molecule.
    """
    standardizer = Standardizer()
    standardizer.prefer_organic = True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        mol = standardizer.standardize(mol)  # standardize functional group reps
        if largest_fragment:
            mol = standardizer.largest_fragment(mol) # remove product counterions/salts/etc.
        mol = standardizer.uncharge(mol)  # neutralize, e.g., carboxylic acids
    except Exception:
        pass
    return Chem.MolToSmiles(mol)

def bookkeep_reactant(mol):
    """Bookkeep bonds in the reactant.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for reactants.

    Returns
    -------
    pair_to_bond_type : dict
        Mapping 2-tuples of atoms to bond type. 1, 2, 3, 1.5 are
        separately for single, double, triple and aromatic bond.
    """
    pair_to_bond_type = dict()
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
        type_val = bond.GetBondTypeAsDouble()
        pair_to_bond_type[(atom1, atom2)] = type_val

    return pair_to_bond_type

bond_change_to_type = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                       3: Chem.rdchem.BondType.TRIPLE, 1.5: Chem.rdchem.BondType.AROMATIC}

clean_rxns_postsani = [
    # two adjacent aromatic nitrogens should allow for H shift
    AllChem.ReactionFromSmarts('[n;H1;+0:1]:[n;H0;+1:2]>>[n;H0;+0:1]:[n;H0;+0:2]'),
    # two aromatic nitrogens separated by one should allow for H shift
    AllChem.ReactionFromSmarts('[n;H1;+0:1]:[c:3]:[n;H0;+1:2]>>[n;H0;+0:1]:[*:3]:[n;H0;+0:2]'),
    AllChem.ReactionFromSmarts('[#7;H0;+:1]-[O;H1;+0:2]>>[#7;H0;+:1]-[O;H0;-:2]'),
    # neutralize C(=O)[O-]
    AllChem.ReactionFromSmarts('[C;H0;+0:1](=[O;H0;+0:2])[O;H0;-1:3]>>[C;H0;+0:1](=[O;H0;+0:2])[O;H1;+0:3]'),
    # turn neutral halogens into anions EXCEPT HCl
    AllChem.ReactionFromSmarts('[I,Br,F;H1;D0;+0:1]>>[*;H0;-1:1]'),
    # inexplicable nitrogen anion in reactants gets fixed in prods
    AllChem.ReactionFromSmarts('[N;H0;-1:1]([C:2])[C:3]>>[N;H1;+0:1]([*:2])[*:3]'),
]

def edit_mol(rmol, bond_changes, keep_atom_map=False):
    """Simulate reaction via graph editing

    Parameters
    ----------
    rmol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants
    bond_changes : list of 3-tuples
        Each tuple is of form (atom1, atom2, change_type)
    keep_atom_map : bool
        Whether to keep atom mapping number. Default to False.

    Returns
    -------
    pred_smiles : list of str
        SMILES for the edited molecule
    """
    new_mol = Chem.RWMol(rmol)

    # Keep track of aromatic nitrogens, which might cause explicit hydrogen issues
    aromatic_nitrogen_ids = set()
    aromatic_carbonyl_adj_to_aromatic_nh = dict()
    aromatic_carbondeg3_adj_to_aromatic_nh0 = dict()
    for atom in new_mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'N':
            aromatic_nitrogen_ids.add(atom.GetIdx())
            for nbr in atom.GetNeighbors():
                if atom.GetNumExplicitHs() == 1 and nbr.GetSymbol() == 'C' and \
                    nbr.GetIsAromatic() and \
                    any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds()):
                    aromatic_carbonyl_adj_to_aromatic_nh[nbr.GetIdx()] = atom.GetIdx()
                elif atom.GetNumExplicitHs() == 0 and nbr.GetSymbol() == 'C' and \
                        nbr.GetIsAromatic() and len(nbr.GetBonds()) == 3:
                    aromatic_carbondeg3_adj_to_aromatic_nh0[nbr.GetIdx()] = atom.GetIdx()
        else:
            atom.SetNumExplicitHs(0)
    new_mol.UpdatePropertyCache()

    for atom1_id, atom2_id, change_type in bond_changes:
        bond = new_mol.GetBondBetweenAtoms(atom1_id, atom2_id)
        atom1 = new_mol.GetAtomWithIdx(atom1_id)
        atom2 = new_mol.GetAtomWithIdx(atom2_id)
        if bond is not None:
            new_mol.RemoveBond(atom1_id, atom2_id)

            # Are we losing a bond on an aromatic nitrogen?
            if bond.GetBondTypeAsDouble() == 1.0:
                if atom1_id in aromatic_nitrogen_ids:
                    if atom1.GetTotalNumHs() == 0:
                        atom1.SetNumExplicitHs(1)
                    elif atom1.GetFormalCharge() == 1:
                        atom1.SetFormalCharge(0)
                elif atom2_id in aromatic_nitrogen_ids:
                    if atom2.GetTotalNumHs() == 0:
                        atom2.SetNumExplicitHs(1)
                    elif atom2.GetFormalCharge() == 1:
                        atom2.SetFormalCharge(0)

            # Are we losing a c=O bond on an aromatic ring?
            # If so, remove H from adjacent nH if appropriate
            if bond.GetBondTypeAsDouble() == 2.0:
                both_aromatic_nh_ids = [
                    aromatic_carbonyl_adj_to_aromatic_nh.get(atom1_id, None),
                    aromatic_carbonyl_adj_to_aromatic_nh.get(atom2_id, None)
                ]
                for aromatic_nh_id in both_aromatic_nh_ids:
                    if aromatic_nh_id is not None:
                        new_mol.GetAtomWithIdx(aromatic_nh_id).SetNumExplicitHs(0)

        if change_type > 0:
            new_mol.AddBond(atom1_id, atom2_id, bond_change_to_type[change_type])

            # Special alkylation case?
            if change_type == 1:
                if atom1_id in aromatic_nitrogen_ids:
                    if atom1.GetTotalNumHs() == 1:
                        atom1.SetNumExplicitHs(0)
                    else:
                        atom1.SetFormalCharge(1)
                elif atom2_id in aromatic_nitrogen_ids:
                    if atom2.GetTotalNumHs() == 1:
                        atom2.SetNumExplicitHs(0)
                    else:
                        atom2.SetFormalCharge(1)

            # Are we getting a c=O bond on an aromatic ring?
            # If so, add H to adjacent nH0 if appropriate
            if change_type == 2:
                both_aromatic_nh0_ids = [
                    aromatic_carbondeg3_adj_to_aromatic_nh0.get(atom1_id, None),
                    aromatic_carbondeg3_adj_to_aromatic_nh0.get(atom2_id, None)
                ]
                for aromatic_nh0_id in both_aromatic_nh0_ids:
                    if aromatic_nh0_id is not None:
                        new_mol.GetAtomWithIdx(aromatic_nh0_id).SetNumExplicitHs(1)

    pred_mol = new_mol.GetMol()

    # Clear formal charges to make molecules valid
    # Note: because S and P (among others) can change valence, be more flexible
    for atom in pred_mol.GetAtoms():
        if not keep_atom_map:
            atom.ClearProp('molAtomMapNumber')
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:
            # exclude negatively-charged azide
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N' and atom.GetFormalCharge() == -1:
            # handle negatively-charged azide addition
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 3 and any([nbr.GetSymbol() == 'N' for nbr in atom.GetNeighbors()]):
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 4 and not atom.GetIsAromatic():
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'C' and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'O' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) + \
                        atom.GetNumExplicitHs()
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ['Cl', 'Br', 'I', 'F'] and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 1:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'P':
            # quartenary phosphorous should be pos. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) == 3 and len(bond_vals) == 3:
                # make sure neutral
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'B':
            # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ['Mg', 'Zn']:
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 1 and len(bond_vals) == 1:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'Si':
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == len(bond_vals):
                atom.SetNumExplicitHs(max(0, 4 - len(bond_vals)))

    # Bounce to/from SMILES to try to sanitize
    pred_smiles = Chem.MolToSmiles(pred_mol)
    pred_list = pred_smiles.split('.')
    pred_mols = [Chem.MolFromSmiles(pred_smiles) for pred_smiles in pred_list]

    for i, mol in enumerate(pred_mols):
        if mol is None:
            continue

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is None:
            continue
        for rxn in clean_rxns_postsani:
            out = rxn.RunReactants((mol,))
            if out:
                try:
                    Chem.SanitizeMol(out[0][0])
                    pred_mols[i] = Chem.MolFromSmiles(Chem.MolToSmiles(out[0][0]))
                except Exception as e:
                    pass

    pred_smiles = [Chem.MolToSmiles(pred_mol) for pred_mol in pred_mols if pred_mol is not None]

    return pred_smiles

def examine_topk_candidate_product(topks, topk_combos, reactant_mol,
                                   real_bond_changes, product_mol):
    """Perform topk evaluation for predicting the product of a reaction

    Parameters
    ----------
    topks : list of int
        Options for top-k evaluation, e.g. [1, 3, ...].
    topk_combos : list of list
        topk_combos[i] gives the combo of valid bond changes ranked i-th,
        which is a list of 3-tuples. Each tuple is of form
        (atom1, atom2, change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. The change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond or
        forming a single, double, triple, aromatic bond.
    reactant_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants.
    real_bond_changes : list of tuples
        Ground truth bond changes in a reaction. Each tuple is of form (atom1, atom2,
        change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
        a single, double, triple, and aromatic bond.
    product_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the product.
    get_smiles : bool
        Whether to get the SMILES of candidate products.

    Returns
    -------
    found_info : dict
        Binary values indicating whether we can recover the product from the ground truth
        graph edits or top-k predicted edits
    """
    found_info = defaultdict(bool)

    # Avoid corrupting the RDKit molecule instances in the dataset
    reactant_mol = deepcopy(reactant_mol)
    product_mol = deepcopy(product_mol)

    for atom in product_mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    product_smiles = Chem.MolToSmiles(product_mol)
    product_smiles_sanitized = set(sanitize_smiles_molvs(product_smiles, True).split('.'))
    product_smiles = set(product_smiles.split('.'))

    ########### Use *true* edits to try to recover product
    # Generate product by modifying reactants with graph edits
    pred_smiles = edit_mol(reactant_mol, real_bond_changes)
    pred_smiles_sanitized = set(sanitize_smiles_molvs(smiles) for smiles in pred_smiles)
    pred_smiles = set(pred_smiles)

    if not product_smiles <= pred_smiles:
        # Try again with kekulized form
        Chem.Kekulize(reactant_mol)
        pred_smiles_kek = edit_mol(reactant_mol, real_bond_changes)
        pred_smiles_kek = set(pred_smiles_kek)
        if not product_smiles <= pred_smiles_kek:
            if product_smiles_sanitized <= pred_smiles_sanitized:
                print('\nwarn: mismatch, but only due to standardization')
                found_info['ground_sanitized'] = True
            else:
                print('\nwarn: could not regenerate product {}'.format(product_smiles))
                print('sani product: {}'.format(product_smiles_sanitized))
                print(Chem.MolToSmiles(reactant_mol))
                print(Chem.MolToSmiles(product_mol))
                print(real_bond_changes)
                print('pred_smiles: {}'.format(pred_smiles))
                print('pred_smiles_kek: {}'.format(pred_smiles_kek))
                print('pred_smiles_sani: {}'.format(pred_smiles_sanitized))
        else:
            found_info['ground'] = True
            found_info['ground_sanitized'] = True
    else:
        found_info['ground'] = True
        found_info['ground_sanitized'] = True

    ########### Now use candidate edits to try to recover product
    max_topk = max(topks)
    current_rank = 0
    correct_rank = max_topk + 1
    sanitized_correct_rank = max_topk + 1
    candidate_smiles_list = []
    candidate_smiles_sanitized_list = []

    for i, combo in enumerate(topk_combos):
        prev_len_candidate_smiles = len(set(candidate_smiles_list))

        # Generate products by modifying reactants with predicted edits.
        candidate_smiles = edit_mol(reactant_mol, combo)
        candidate_smiles = set(candidate_smiles)
        candidate_smiles_sanitized = set(sanitize_smiles_molvs(smiles)
                                         for smiles in candidate_smiles)

        if product_smiles_sanitized <= candidate_smiles_sanitized:
            sanitized_correct_rank = min(sanitized_correct_rank, current_rank + 1)
        if product_smiles <= candidate_smiles:
            correct_rank = min(correct_rank, current_rank + 1)

        # Record unkekulized form
        candidate_smiles_list.append('.'.join(candidate_smiles))
        candidate_smiles_sanitized_list.append('.'.join(candidate_smiles_sanitized))

        # Edit molecules with reactants kekulized. Sometimes previous editing fails due to
        # RDKit sanitization error (edited molecule cannot be kekulized)
        try:
            Chem.Kekulize(reactant_mol)
        except Exception as e:
            pass

        candidate_smiles = edit_mol(reactant_mol, combo)
        candidate_smiles = set(candidate_smiles)
        candidate_smiles_sanitized = set(sanitize_smiles_molvs(smiles)
                                         for smiles in candidate_smiles)
        if product_smiles_sanitized <= candidate_smiles_sanitized:
            sanitized_correct_rank = min(sanitized_correct_rank, current_rank + 1)
        if product_smiles <= candidate_smiles:
            correct_rank = min(correct_rank, current_rank + 1)

        # If we failed to come up with a new candidate, don't increment the counter!
        if len(set(candidate_smiles_list)) > prev_len_candidate_smiles:
            current_rank += 1

        if correct_rank < max_topk + 1 and sanitized_correct_rank < max_topk + 1:
            break

    for k in topks:
        if correct_rank <= k:
            found_info['top_{:d}'.format(k)] = True
        if sanitized_correct_rank <= k:
            found_info['top_{:d}_sanitized'.format(k)] = True

    return found_info

def summary_candidate_ranking_info(top_ks, found_info, data_size):
    """Get a string for summarizing the candidate ranking results

    Parameters
    ----------
    top_ks : list of int
        Options for top-k evaluation, e.g. [1, 3, ...].
    found_info : dict
        Storing the count of correct predictions
    data_size : int
        Size for the dataset

    Returns
    -------
    string : str
        String summarizing the evaluation results
    """
    string = '[strict]'
    for k in top_ks:
        string += ' acc@{:d}: {:.4f}'.format(k, found_info['top_{:d}'.format(k)] / data_size)
    string += ' gfound {:.4f}\n'.format(found_info['ground'] / data_size)
    string += '[molvs]'
    for k in top_ks:
        string += ' acc@{:d}: {:.4f}'.format(
            k, found_info['top_{:d}_sanitized'.format(k)] / data_size)
    string += ' gfound {:.4f}\n'.format(found_info['ground_sanitized'] / data_size)

    return string

def candidate_ranking_eval(args, model, data_loader):
    """Evaluate model performance on candidate ranking.

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
        String summarizing the evaluation results
    """
    model.eval()

    # Record how many product can be recovered by real graph edits (with/without sanitization)
    found_info_summary = {'ground': 0, 'ground_sanitized': 0}
    for k in args['top_ks']:
        found_info_summary['top_{:d}'.format(k)] = 0
        found_info_summary['top_{:d}_sanitized'.format(k)] = 0

    total_samples = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactant_graphs, batch_product_graphs, batch_combo_scores, \
        batch_valid_candidate_combos, batch_reactant_mols, batch_real_bond_changes, \
        batch_product_mols, batch_num_candidate_products = batch_data

        # No valid candidate products have been predicted
        if batch_reactant_graphs is None:
            continue
        total_samples += len(batch_num_candidate_products)

        batch_combo_scores = batch_combo_scores.to(args['device'])
        reactant_node_feats = batch_reactant_graphs.ndata.pop('hv').to(args['device'])
        reactant_edge_feats = batch_reactant_graphs.edata.pop('he').to(args['device'])
        product_node_feats = batch_product_graphs.ndata.pop('hv').to(args['device'])
        product_edge_feats = batch_product_graphs.edata.pop('he').to(args['device'])

        # Get candidate products with top-k ranking
        with torch.no_grad():
            pred = model(reactant_graph=batch_reactant_graphs,
                         reactant_node_feats=reactant_node_feats,
                         reactant_edge_feats=reactant_edge_feats,
                         product_graphs=batch_product_graphs,
                         product_node_feats=product_node_feats,
                         product_edge_feats=product_edge_feats,
                         candidate_scores=batch_combo_scores,
                         batch_num_candidate_products=batch_num_candidate_products)

        product_graph_start = 0
        for i in range(len(batch_num_candidate_products)):
            num_candidate_products = batch_num_candidate_products[i]
            reactant_mol = batch_reactant_mols[i]
            valid_candidate_combos = batch_valid_candidate_combos[i]
            real_bond_changes = batch_real_bond_changes[i]
            product_mol = batch_product_mols[i]

            product_graph_end = product_graph_start + num_candidate_products
            top_k = min(args['max_k'], num_candidate_products)
            reaction_pred = pred[product_graph_start:product_graph_end, :]
            topk_values, topk_indices = torch.topk(reaction_pred, top_k, dim=0)

            # Filter out invalid candidate bond changes
            reactant_pair_to_bond = bookkeep_reactant(reactant_mol)
            topk_combos = []
            for i in topk_indices:
                i = i.detach().cpu().item()
                combo = []
                for atom1, atom2, change_type, score in valid_candidate_combos[i]:
                    bond_in_reactant = reactant_pair_to_bond.get((atom1, atom2), None)
                    if (bond_in_reactant is None and change_type > 0) or \
                            (bond_in_reactant is not None and bond_in_reactant != change_type):
                        combo.append((atom1, atom2, change_type))
                topk_combos.append(combo)

            batch_found_info = examine_topk_candidate_product(
                args['top_ks'], topk_combos, reactant_mol, real_bond_changes, product_mol)
            for k, v in batch_found_info.items():
                found_info_summary[k] += float(v)

            product_graph_start = product_graph_end

        if total_samples % args['print_every'] == 0:
            print('Iter {:d}/{:d}'.format(
                total_samples // args['print_every'],
                len(data_loader.dataset) // args['print_every']))
            print(summary_candidate_ranking_info(
                args['top_ks'], found_info_summary, total_samples))

    return summary_candidate_ranking_info(args['top_ks'], found_info_summary, total_samples)
